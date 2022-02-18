## Architecture

![Untitled](/image/Untitled.png){:width="400"}

- Image Classification을 하는 모델
- Encoder만 사용
- 첫번째 encoded vector를 바로 classifier에 넣는 구조

1. image의 sub-patch들을 1-d embedding (flatten)으로 만든다.
    
    $$
    x∈\mathbb{R}^{H*W*C} → x_p∈\mathbb{R}^{N*(P^2*C)}
    $$
    
    - $H$ : height, $W$ : width, $C$ : channel 수
    - $N$ : patch 수, $P$ : 한 patch당 size
2. linear projection of flatten patches
    - $D$ : 모든 layer에서의 latent vector size (embedding vector) (c*p*p)
3. class token + positional embedding
    - class token
        - 전체 이미지의 representation을 나타내는 특별한 token
        - input sequence **맨 앞**에 learnable embedding parameter를 **concat**한다.
        - 최종 classification head에서 사용한다.
    - Positional Embedding
        - **각 patch의 순서**를 알려주는 역할
        - 학습되는 파라미터
4. Encoder n번 수행
    - Layer Normalization
    - multi-head self-attention
    - residual connection
    - MLP
5. MLP 통해 classification 수행

<br/>

## Image Embedding 구현 코드

### labml 구현

```python
class LearnedPositionalEmbeddings(Module):
	def __init__(self, d_model: int, max_len: int = 5_000):
		# d_model : transformer embeddings size (c*p*p)
		# max_len : maximum number of patches

		super().__init__()

		# dimension : max_len * 1 * d_model
		# 의문점 : max_len + 1, d_model 아닌가?
		self.positional_embeddings = nn.Parameter(torch.zeros(max_len, 1, d_model), 
																							requires_grad=True)

	def forward(self, x: torch.Tensor):
		pe = self.positional_embeddings[x.shape[0]]  # batch 크기만큼 가져와 더하기
		return x + pe
```

### 심화과제 구현

- 참고 사이트
    
    [친절한 코드 해석](https://yhkim4504.tistory.com/5)
    

```python
class image_embedding(nn.Module):
  def __init__(self, in_channels: int = 3, img_size: int = 224, patch_size: int = 16, 
								emb_dim: int = 16*16*3):
    super().__init__()

    ### [B, C, H, W] → [B, N, (c * P^2)]
		# einops
    self.rearrange = Rearrange('b c (num_h p1) (num_w p2) -> b (num_h num_w) (c p1 p2)', 
																p1=patch_size, p2=patch_size)
    
    ### linear projection
    self.linear = nn.Linear(in_channels * patch_size * patch_size, emb_dim)

    ### class token 초기화 (image의 차원이 3가지, '맨 앞'에 붙여줌)
    self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
    
    ### positional embedding 초기화 ('patch 수 + class token'에 더해줌)
    n_patches = img_size * img_size // patch_size**2
    self.positions = nn.Parameter(torch.randn(n_patches + 1, emb_dim))

  def forward(self, x):
    batch, channel, height, width = x.shape

    x = self.rearrange(x) # flatten patches
    x = self.linear(x) # embedded patches

    # cls_token을 반복하여 배치사이즈의 크기와 맞춰줌 (각 batch 단위에 class token을 concat)
    c = repeat(self.cls_token, '() n d -> b n d', b=batch)
    x = torch.cat((c, x), dim=1)
    x = x + self.positions
   
    return x

emb = image_embedding(1, 28, 4, 1*4*4)(x)
emb.shape

"""
torch.Size([1, 49, 16])
torch.Size([1, 50, 16])
torch.Size([1, 50, 16])
"""
```