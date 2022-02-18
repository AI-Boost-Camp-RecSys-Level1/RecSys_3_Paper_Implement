## ResNet

- 굉장히 deep한데 (**128 layers**) 성능이 좋다!
- **Degradation** 문제를 해결하기 위해 **residual function**을 사용한다.
    - Degradation? DNN의 layer가 증가할수록 accuracy가 감소(degrade)하는 현상

<br/>

## Plain Network의 한계점

- (ResNet의 핵심 아이디어인) skip connection이 없는 CNN
- layer가 깊어질수록...
    - **Gradient Vanishing/Exploding** 발생
        - parameter를 업데이트하기 위해 gardient descent를 하게 되는데, 이 때 gradient값이 너무 작거나 큰 값들이 계속 곱해지면 결국 vanishig 혹은 exploding하게 된다.
    - layer를 너무 많이 쌓으면 **training error가 높게 나타난다. → 결국 degradation!**
        
        ![Untitled](/images/1.png){:width="400"}
        
<br/>

## Residual Learning

이 블록은 입력 x와 실제 분포(true distribution) H(x)를 학습한다. 아래처럼 모델의 입력과 출력간의 차이(또는 잔차)를 표시해 보자.

- **Residual** Function $F(x)$
    - 모델의 output funciton $H(x)$ - 모델의 input $x$
    
    $$
    F(x)=H(x)-x \quad → \quad  H(x)=F(x)+x
    $$
    
    - 결국 convolution layer를 거친 값
    - Neural Network의 목적 : 정답을 잘 예측하는 것.
        - 다시 말해 $H(x)$가 $x$로 나오게 끔 하는 것이 목표!
        - **residual function $F(x)$가 0이 되도록 학습한다!**

<br/>

## Skip (Shortcut) Connection

![Untitled](/images/2.png){:width="400"}

$$
H(x) = F(x)+x
$$

- **입력 $x$를 몇 layer 이후의 output에 더해준다.**
- output에서의 gradient를 구하면 $\frac{\partial H}{\partial x}=\frac{\partial F}{\partial x}+1$ 이 되므로 $\frac{\partial H}{\partial x}$이 0이 되지 않아 **Gradient Vanishing/Exploding 문제를 해결**할 수 있다.

<br/>

## Architecture

![archi.png](/images/3.png){:width="400"}

- **위**에서부터 순서대로 **residual net**, plain net, VGG-19
    - ResNet은 VGGnet 구조에서 가져온 것이다.
- skip connection의 조건 : **$x$의 size = output의 size**
    
    $$
    H(x) = F(x, \{Wi\}) + x,\quad   F(x, \{Wi\}) = W_2\;σ(W_1x)
    $$
    
    - 만약 size가 동일하지 않다면 **linear projection $W_s$**를 적용할 수 있다.
        
        $$
        H(x)=F(x,\{W_i\})+W_s x
        $$
        
- **input size < output size**일 때 사용하는 3가지의 skip connection
    - **증가하는 차원**에 대해 추가적으로 **zero padding**을 적용하여 identity mapping 수행
    - 차원이 **증가할 때만** **projection shortcut** 사용
    - **모든** **shortcut**이 **projection**

<br/>

## class ResidualBlock

[labml 구현](https://nn.labml.ai/resnet/index.html)

![Untitled](/images/4.png){:height="500"}

- 기본적으로 CNN 구조
- 입력 $x$(이전 단계의 output)를 2개 convolution layer 이후의 output에 더해준다.
- 1st 3*3 Conv layer
- 2nd 3*3 Conv layer
- Shortcut Projection