import os
import random
import logging
import itertools
import numpy as np # type: ignore
import torch # type: ignore
from torch.utils.data import DataLoader # type: ignore
from datasets.dataset import DrumDataset
from models.gan import Generator, Conditioner, Discriminator, weights_init_normal
from utilities.train import train
from flask import Flask, request, jsonify # type: ignore
from flask_cors import CORS # type: ignore
from dotenv import load_dotenv # type: ignore

load_dotenv()
EXPECTED_API_KEY = os.getenv("API_KEY", "demo_key")

# Configure logging for Cloud Run / Cloud Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

class Options:
    def __init__(self):
        self.min_pitch = int(os.getenv("MIN_PITCH", "22"))
        self.max_pitch = int(os.getenv("MAX_PITCH", "59"))
        self.w = int(os.getenv("W", "16"))
        self.latent_dim = int(os.getenv("LATENT_DIM", "100"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "32"))
        self.lr = float(os.getenv("LEARNING_RATE", "0.0002"))
        self.ckpt_dir = os.getenv("CKPT_DIR", "checkpoints/")
        self.pretrained = os.getenv("PRETRAINED", "weights/")
        self.pin_memory = os.getenv("PIN_MEMORY", "False").lower() in {"1", "true", "yes", "on"}
        self.genre = os.getenv("GENRE", "all")
        self.epochs = int(os.getenv("EPOCHS", "10"))
        self.num_workers = int(os.getenv("NUM_WORKERS", "4"))
        self.lambda_1 = float(os.getenv("LAMBDA_1", "0.01"))
        self.lambda_2 = float(os.getenv("LAMBDA_2", "0.1"))
        self.fine_tune = os.getenv("FINE_TUNE", "False").lower() in {"1", "true", "yes", "on"}
    
@app.route("/", methods=["GET"])
def health_check():
    """Health check route for container deployment validation."""
    return jsonify({"status": "healthy", "service": "MIDInet Training Service"}), 200


@app.route("/api/train", methods=["POST"])
def trigger_training_pipeline():
    """
    Endpoint triggered by Google Cloud Scheduler or authorized admin requests 
    to execute background training and upload weights to Google Cloud Storage.
    """
    # Example Payload expectation JSON:
    # {
    #     "genre": "rock",
    #     "epochs": 10,
    #     "batch_size": 32,
    #     "learning_rate": 0.0002,
    #     "latent_dim": 100,
    #     "min_pitch": 22,
    #     "max_pitch": 59,
    #     "w": 16,
    #     "num_workers": 4,
    #     "pin_memory": false,
    #     "lambda_1": 0.01,
    #     "lambda_2": 0.1,
    #     "ckpt_dir": "checkpoints/",
    #     "pretrained": "weights/"
    # } 
    # 1. Validate API Key / Authorization Header
    auth_header = request.headers.get("X-API-Key")
    if not auth_header or auth_header != EXPECTED_API_KEY:
        logger.warning("Unauthorized training trigger attempt detected.")
        return jsonify({"error": "Unauthorized access"}), 401

    # 2. Extract incoming parameters (e.g., from Cloud Scheduler job payload)
    opt = Options()
    
    req_data = request.get_json() or {}
    
    opt.genre = req_data.get("genre", "all")
    opt.epochs = int(req_data.get("epochs", 10))
    opt.batch_size = int(req_data.get("batch_size", 32))
    opt.lr = float(req_data.get("learning_rate", 0.0002))
    opt.latent_dim = int(req_data.get("latent_dim", 100))
    opt.min_pitch = int(req_data.get("min_pitch", 22))
    opt.max_pitch = int(req_data.get("max_pitch", 59))
    opt.w = int(req_data.get("w", 16))
    opt.num_workers = int(req_data.get("num_workers", 4))
    opt.pin_memory = req_data.get("pin_memory", False)
    opt.lambda_1 = float(req_data.get("lambda_1", 0.01))
    opt.lambda_2 = float(req_data.get("lambda_2", 0.1))
    opt.ckpt_dir = req_data.get("ckpt_dir", "checkpoints/")
    opt.pretrained = req_data.get("pretrained", "weights/")
    opt.fine_tune = req_data.get("fine_tune", False)
    
    # Set seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.info(f"Starting training pipeline for genre: {opt.genre} for {opt.epochs} epochs...")

    try:
        
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Training on: {device}")
        
        generator = Generator(min_pitch=opt.min_pitch, max_pitch=opt.max_pitch, w=opt.w, latent_dim=opt.latent_dim, chord_dim=0)
        conditioner = Conditioner(min_pitch=opt.min_pitch, max_pitch=opt.max_pitch, w=opt.w)
        discriminator = Discriminator(min_pitch=opt.min_pitch, max_pitch=opt.max_pitch, w=opt.w, chord_dim=0)
    
        generator.to(device)
        conditioner.to(device)
        discriminator.to(device)
    
        print(f"Initializing dataset and dataloaders with batch size {opt.batch_size} for genre type: {opt.genre}...")
        dataset = DrumDataset(
            metadata_path="./datasets/metadata.json", 
            genre_filter=opt.genre, 
            device=device
        )
        
        # Create train and evaluation splits (80% train, 20% eval)
        dataset_size = len(dataset)
        if dataset_size == 0:
            raise ValueError(f"No data found for genre '{opt.genre}' in the dataset.")
        train_size = int(0.8 * dataset_size)
        eval_size = dataset_size - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

        if device.type == "cuda":
            loader_kwargs = {
                "num_workers": 0,
                "pin_memory": opt.pin_memory
            }
        else:
            loader_kwargs = {
                "num_workers": 0,
                "pin_memory": False
            }
        print("Loading DataLoaders")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            **loader_kwargs
        )
    
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            **loader_kwargs
        )
        
        print(f"Training dataset size: {len(train_dataset)}, Evaluation dataset size: {len(eval_dataset)}")
        os.makedirs(opt.ckpt_dir, exist_ok=True)
    
        g_c_params = itertools.chain(generator.parameters(), conditioner.parameters())
        generator_optimizer = torch.optim.RMSprop(g_c_params, lr=opt.lr)
        discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)
    
        if opt.fine_tune:
            model_path = os.path.join(opt.pretrained, f"{opt.genre}_model_weights.pth")
            checkpoint = torch.load(model_path, map_location=device)
            generator.load_state_dict(checkpoint["generator_state_dict"])
            conditioner.load_state_dict(checkpoint["conditioner_state_dict"])
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        
        generator.train()
        discriminator.train()
        conditioner.train()
    
        print(f"Batch Info: {opt.batch_size} batch size, {len(train_dataloader)} batches per epoch")
    
        train(
            generator,
            conditioner,
            discriminator,
            train_dataloader,
            eval_dataloader,
            generator_optimizer,
            discriminator_optimizer,
            opt,
            device,
            checkpoint_dir=opt.ckpt_dir
        )
    
        possible_models = [f for f in os.listdir(opt.ckpt_dir) if f.endswith(".pth") and f.startswith(opt.genre)]
        if not possible_models:
            raise FileExistsError("No saved models found in checkpoints directory.")
        for model in possible_models:
            try:
                checkpoint = torch.load(os.path.join(opt.ckpt_dir, model), map_location=device)
            except Exception as e:
                logger.error(f"Failed to load model {model}: {str(e)}")
                continue
            
        logger.info("Training completed successfully and weights stored to GCS.")
        
        return jsonify({
            "status": "success",
            "message": f"Training completed successfully for genre '{opt.genre}'.",
            "epochs_run": opt.epochs
        }), 200

    except Exception as e:
        logger.error(f"Training pipeline failed with error: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=True)