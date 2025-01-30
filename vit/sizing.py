from torchsummary import summary
from model import ViT, ViTConfig
config = ViTConfig(
    image_size=(224, 224),
    patch_size=(16, 16),
    num_classes=100,
    dim=512,
    dim_head=64,
    depth=12,
    num_heads=8,
    hidden_dim=2048,
    channels=3,
    pool='cls',
    dropout=0.1,
    emb_dropout=0.1
)
model = ViT(config).to('cuda')
summary(model, (3, 224, 224), device='cuda')
