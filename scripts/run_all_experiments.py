import sys
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import Config
from src.model import get_model_factory
from dataset.loaders import get_dataloaders
from scripts.tta import evaluate_with_tta, plot_tta_metrics
from scripts.visualize import generate_cams, save_text_as_png, run_statistical_tests
from scripts.calibration import run_calibration
import wandb

def main():
    print("Iniciando bateria de experimentos automatizados...")
    
    model_names = ["densenet121", "resnet50", "efficientnet_b2", "inception_v3", "convnext_tiny"]
    
    # Grid de experimentos: Modelos x (SEG vs FULL)
    experimentos = []
    for m in model_names:
        experimentos.append({"model": m, "is_seg": True, "use_gu": True})
        experimentos.append({"model": m, "is_seg": False, "use_gu": True})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for exp in experimentos:
        # 1. Sobrescreve as configurações globalmente
        Config.MODEL_NAME = exp["model"]
        Config.USE_SEGMENTED_DATA = exp["is_seg"]
        Config.USE_GRADUAL_UNFREEZING = exp["use_gu"]
        Config.EXPERIMENT_NAME = f"{exp['model']}_{'SEG' if exp['is_seg'] else 'FULL'}_{'GU' if exp['use_gu'] else 'NOGU'}"
        
        Config.IMG_OUTPUTS_PATH = os.path.join(os.getcwd(), 'outputs', Config.EXPERIMENT_NAME)
        
        # Super-Skip: Se o checkpoint e o último gráfico já existem, a avaliação inteira já foi concluída.
        import glob
        checkpoint_dir = os.path.join(Config.BASE_PATH, 'checkpoints', Config.EXPERIMENT_NAME)
        already_evaluated = os.path.exists(os.path.join(Config.IMG_OUTPUTS_PATH, "roc_curve_comparison.png"))
        
        if already_evaluated and len(glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))) > 0:
            print(f"\n[Super Skip] Treino e Avaliação já 100% concluídos para {Config.EXPERIMENT_NAME}. Pulando experimento inteiro!")
            continue

        os.makedirs(Config.IMG_OUTPUTS_PATH, exist_ok=True)
        
        print(f"\n{'='*50}\nIniciando Experimento: {Config.EXPERIMENT_NAME}\n{'='*50}")
        
        # 2. Recarrega os Datasets de acordo com as novas regras (Segmentado ou Full)
        train_loader, val_loader, test_loader = get_dataloaders()
        
        # 3. Prepara o Logger e o Modelo
        wandb_logger = WandbLogger(
            project="SIATCT_Binary_Experiment",
            name=Config.EXPERIMENT_NAME,
            log_model="all"
        )
        
        model = get_model_factory(
            model_name=exp["model"],
            num_classes=Config.NUM_CLASSES, 
            learning_rate=Config.LEARNING_RATE
        )
        
        # Callbacks (salva os pesos na pasta com o nome do experimento atual)
        checkpoint_dir = os.path.join(Config.BASE_PATH, 'checkpoints', Config.EXPERIMENT_NAME)
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )
        
        # Verifica se já existe um modelo treinado para pular
        import glob
        existing_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        already_trained = len(existing_checkpoints) > 0
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # 4. Treinamento
        # Obs: Descomente os limites caso queira rodar um teste rápido (ex: 10% dos dados)
        trainer = Trainer(
            max_epochs=Config.MAX_EPOCHS,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, lr_monitor],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="16-mixed",
            # limit_train_batches=0.1,  # Para testes rápidos
            # limit_val_batches=0.1,
        )
        
        if already_trained:
            print(f"\n[Smart Skip] Checkpoint encontrado para {Config.EXPERIMENT_NAME}. Pulando treino!")
            best_model_path = existing_checkpoints[0]
        else:
            trainer.fit(model, train_loader, val_loader)
            
            # 5. Avaliação Final (Pós-Treino)
            print(f"\nTreino concluído para {Config.EXPERIMENT_NAME}. Iniciando análise rigorosa...")
            
            # Carrega o melhor peso gerado
            best_model_path = checkpoint_callback.best_model_path
            if not best_model_path:
                best_model_path = checkpoint_callback.last_model_path
            
        print(f"Avaliando pesos: {best_model_path}")
        best_model = model.__class__.load_from_checkpoint(best_model_path)
        best_model.to(device)
        best_model.eval()
        
        # Gera o TTA e métricas de texto em PNG
        # OBS: evaluate_with_tta do script tta.py deverá retornar o classification report como string,
        # ou você pode acionar save_text_as_png passando as saídas.
        print("Executando TTA e plotando Curvas...")
        results = evaluate_with_tta(best_model, test_loader, device)
        plot_tta_metrics(results)
        
        # Testes Estatísticos (Bootstrapping, CIs, McNemar preds)
        print("Executando Testes Estatísticos...")
        run_statistical_tests(results, Config.EXPERIMENT_NAME)
        
        # Gera e salva os Grad-CAMs
        print("Gerando Explicabilidade (Grad-CAM, ++, HiResCAM)...")
        generate_cams(best_model, test_loader, device, num_images=4)
        
        # Testes de Calibração (ECE e Temperature Scaling)
        print("Executando Análise de Calibração (ECE)...")
        run_calibration(best_model, val_loader, test_loader, device)
        
        # Envia TODAS as imagens geradas nesta rodada para o painel do WandB!
        import glob
        print("Fazendo upload dos gráficos e heatmaps para o WandB...")
        png_files = glob.glob(os.path.join(Config.IMG_OUTPUTS_PATH, "*.png"))
        if png_files:
            wandb_images = {os.path.basename(f).replace('.png', ''): wandb.Image(f) for f in png_files}
            wandb.log(wandb_images)
        
        # Finaliza o logging do wandb para este run e desliga
        wandb.finish()
        print(f"Experimento {Config.EXPERIMENT_NAME} 100% Finalizado!\n")

if __name__ == "__main__":
    main()
