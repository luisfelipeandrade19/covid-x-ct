import sys
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import Config
from src.model import SimpleClassifier
from dataset.loaders import get_dataloaders
from scripts.tta import evaluate_with_tta
from scripts.visualize import generate_cams, save_text_as_png, run_statistical_tests
import wandb

def main():
    print("Iniciando bateria de experimentos automatizados...")
    
    # Grid Search sobre as duas variáveis-chave do experimento
    experimentos = [
        {"is_seg": True, "use_gu": True},
        {"is_seg": True, "use_gu": False},
        {"is_seg": False, "use_gu": True},
        {"is_seg": False, "use_gu": False},
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for exp in experimentos:
        # 1. Sobrescreve as configurações globalmente
        Config.USE_SEGMENTED_DATA = exp["is_seg"]
        Config.USE_GRADUAL_UNFREEZING = exp["use_gu"]
        Config.EXPERIMENT_NAME = f"{'SEG' if exp['is_seg'] else 'FULL'}_{'GU' if exp['use_gu'] else 'NOGU'}"
        
        # Atualiza o caminho de output dinâmico para essa iteração
        Config.IMG_OUTPUTS_PATH = os.path.join(os.getcwd(), 'outputs', Config.EXPERIMENT_NAME)
        os.makedirs(Config.IMG_OUTPUTS_PATH, exist_ok=True)
        
        print(f"\n{'='*50}\nIniciando Experimento: {Config.EXPERIMENT_NAME}\n{'='*50}")
        
        # 2. Recarrega os Datasets de acordo com as novas regras (Segmentado ou Full)
        train_loader, val_loader, test_loader = get_dataloaders()
        
        # 3. Prepara o Logger e o Modelo
        wandb_logger = WandbLogger(
            project="SIATCT_Binary_Experiment",
            name=f"DenseNet_{Config.EXPERIMENT_NAME}",
            log_model="all"
        )
        
        model = SimpleClassifier(
            num_classes=Config.NUM_CLASSES, 
            learning_rate=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Callbacks (salva os pesos na pasta com o nome do experimento atual)
        checkpoint_dir = os.path.join(Config.BASE_PATH, 'checkpoints', Config.EXPERIMENT_NAME)
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
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
        
        trainer.fit(model, train_loader, val_loader)
        
        # 5. Avaliação Final (Pós-Treino)
        print(f"\nTreino concluído para {Config.EXPERIMENT_NAME}. Iniciando análise rigorosa...")
        
        # Carrega o melhor peso gerado
        best_model_path = checkpoint_callback.best_model_path
        if not best_model_path:
            best_model_path = checkpoint_callback.last_model_path
            
        print(f"Avaliando pesos: {best_model_path}")
        best_model = SimpleClassifier.load_from_checkpoint(best_model_path)
        best_model.to(device)
        best_model.eval()
        
        # Gera o TTA e métricas de texto em PNG
        # OBS: evaluate_with_tta do script tta.py deverá retornar o classification report como string,
        # ou você pode acionar save_text_as_png passando as saídas.
        print("Executando TTA e plotando Curvas...")
        results = evaluate_with_tta(best_model, test_loader, device)
        
        # Testes Estatísticos (Bootstrapping, CIs, McNemar preds)
        print("Executando Testes Estatísticos...")
        run_statistical_tests(results, Config.EXPERIMENT_NAME)
        
        # Gera e salva os Grad-CAMs
        print("Gerando Explicabilidade (Grad-CAM, ++, HiResCAM)...")
        generate_cams(best_model, test_loader, device, num_images=4)
        
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
