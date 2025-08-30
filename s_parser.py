import os
import argparse
import torch


# function to parse boolean args
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
        
        


def parse_args():

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Senseiver")
    
    # Data
    parser.add_argument("--data_name", default='bay', type=str)
    parser.add_argument("--num_sensors", default=0.9, type=float) # 0.1 0.3 0.5 0.7 0.9
    parser.add_argument("--gpu_device", default=0, type=int)
    parser.add_argument("--training_frames", default=4896, type=int)
    parser.add_argument("--consecutive_train", default=True, type=str2bool)
    parser.add_argument("--seed", default=42, type=int) #
    parser.add_argument("--batch_frames", default=128, type=int)
    parser.add_argument("--batch_pixels", default=64, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--accum_grads", default=1, type=int)
    parser.add_argument("--temp_dim_tid", default=32, type=int)
    parser.add_argument("--temp_dim_diw", default=32, type=int)
    parser.add_argument("--temp_dim_wea", default=32, type=int)

    # Positional Encodings
    parser.add_argument("--space_bands", default=32, type=int)
    
    
    # Checkpoints
    parser.add_argument("--load_model_num", default=5, type=int)  # 0-->0.1 , 1-->0.3,3-->0.5,4-->0.7,5-->0.9
    parser.add_argument("--test", default=True, type=str2bool)
    
    # Encoder
    parser.add_argument("--enc_preproc_ch", default=16, type=int) # 8 16 32 
    parser.add_argument("--num_latents", default=128, type=int) # 32 64 128
    parser.add_argument("--enc_num_latent_channels", default=32, type=int) # 8 16 32
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--num_cross_attention_heads", default=2, type=int)
    parser.add_argument("--enc_num_self_attention_heads", default=2, type=int)
    parser.add_argument("--num_self_attention_layers_per_block", default=3, type=int)
    parser.add_argument("--dropout", default=0.00, type=float)
    
    # Decoder
    parser.add_argument("--dec_preproc_ch", default=64, type=int) # 16 32  64
    parser.add_argument("--dec_num_latent_channels", default=32, type=int) # 8 16 32
    parser.add_argument("--dec_num_cross_attention_heads", default=1, type=int)
    
    
    args = parser.parse_args()

    if torch.cuda.is_available():
        accelerator = "gpu"
        gpus = [args.gpu_device]
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        gpus = [args.gpu_device]
    else:
        accelerator = "cpu"
        gpus = None
        
    # Assign the args
    data_config = dict(data_name = args.data_name,
                       num_sensors = args.num_sensors,
                       gpu_device=None if accelerator == 'cpu' else gpus,
                       accelerator = accelerator,
                       training_frames = args.training_frames,
                       consecutive_train = args.consecutive_train,
                       seed = args.seed,
                       batch_frames = args.batch_frames,
                       batch_pixels = args.batch_pixels,
                       lr=args.lr,
                       accum_grads = args.accum_grads,
                       test = args.test,
                       temp_dim_tid = args.temp_dim_tid,
                       temp_dim_diw = args.temp_dim_diw,
                       temp_dim_wea = args.temp_dim_wea,
                       # node_dim  =  args.node_dim,
                       space_bands=args.space_bands,
                       )

    
    
    encoder_config = dict(load_model_num=args.load_model_num,
                          enc_preproc_ch=args.enc_preproc_ch,  # expand input dims
                          num_latents=args.num_latents,     # "seq" latent
                          enc_num_latent_channels=args.enc_num_latent_channels,  # channels [b,seq,chan]
                          num_layers=args.num_layers,
                          num_cross_attention_heads=args.num_cross_attention_heads,
                          enc_num_self_attention_heads=args.enc_num_self_attention_heads,
                          num_self_attention_layers_per_block=args.num_self_attention_layers_per_block,
                          dropout=args.dropout,
                          )


    decoder_config = dict(dec_preproc_ch=args.dec_preproc_ch,  # latent bottleneck
                          dec_num_latent_channels=args.dec_num_latent_channels,  # hyperparam
                          latent_size=1,  # collapse from n_sensors to 1 observation
                          dec_num_cross_attention_heads=args.dec_num_cross_attention_heads
                          )
    
    
    return data_config, encoder_config, decoder_config  
