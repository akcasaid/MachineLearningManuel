import numpy as np
class neural_network:

## Aşağıda basit Sinir Ağı 2 katmandan oluşmaktadır:
#Gizli Katman
#Çıktı Katmanı

#İlk olarak, ağırlıklar ve sapmalarla birlikte katmanların boyutunu başlatın. Ve ayrıca sigmoid aktivasyon fonksiyonunu ve onun türevini tanımlayın, ki bu gerçekten doğrusal olmama durumunu tanıtmak için anahtardır.

    def __init__(self, input_size,hidden_size, outpout_size):
        self.input_size  =  input_size
        self.hidden_size = hidden_size
        self.output_size = outpout_size

        self.W1 = np.random.randn(self.input_size, self.hidden_size)  #weight
        self.b1 = np.zeros((1,self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

        def sigmoid (self, x):
            return (1 /(1+ np.exp(-x)))
        
        def sigmoid_derrivate (self,x):
            return x* (1-x)
        
        #Burada girdi verileri, tahmin edilen çıktıyı elde etmek için sinir ağından geçirilir. İleri geçişte, Önce gizli katmanın çıktısını hesaplayın.

        #gizli_çıktı = X•W1 + b1    Ardından sigmoid aktivasyonunu çıktıya uygulayın.      çıkış = sigmoid ( ( X•W1) + b1)
        
        def forward(self, X):
            self.hidden_output = self.sigmoid(np.dot(X, self.W1) + self.b1 )
            self.output = self.sigmoid(np.dot(self.hidden_output, self.W2) + self.b2)
            return self.output
        

            #Backward Pass:  First compute the gradients of the output layer.  Gradient of Loss = (y - output) * sigmoid_derivative(output)
            # Now calculate d_W2 which is gradient of the loss function with respect to W2.  d_W2 = hidden_output.T • Gradient of Loss
            #dW1: W1'e göre kayıp fonksiyonunun gradyanı
            #d_b2: Kayıp fonksiyonunun b2'ye göre gradyanı (çıktı katmanındaki nöronun önyargısı)
            #d_b1: Kayıp fonksiyonunun b1'e göre gradyanı (gizli katmandaki nöronun önyargısı) 
        
        def backward(self, X, y, learning_rate):
            d_output = (y- self.output) * self.sigmoid_derrivate(self.output)
            d_W2 = np.dot(self.hidden_output.T, d_output) 
            d_b2 = np.sum(d_output, axis=0, keepdims=True)

            d_hidden=np.dot(d_output, self.W2.T) * self.sigmoid_derivate(self.hidden_output)
            d_W1 = np.dot(X.T, d_hidden)
            d_b1 =np.dot(d_hidden, axis=0, keepdims=True)

            #Şimdi Ağırlıkları Güncelleyin:
            #Burada öğrenme oranı hiper parametredir! Düşük bir öğrenme oranı modelin yerel optimalara takılmasına neden olabilirken, yüksek öğrenme oranı modelin genel çözümü aşmasına neden olabilir
            #W1 += öğrenme_hızı * d_W1 b1 += öğrenme_hızı * d_b1 

            self.W2 += learning_rate * d_W2
            self.W2 += learning_rate * d_b2
            self.W1 += learning_rate * d_W1
            self.W1 += learning_rate * d_b1


            def train (self,X,y,epoch, learning_rate):
                for epoch in range(epoch):
                    output = self.forward(X)

                    self.backward(X,y , learning_rate)
                    loss = np.mean((y- output) ** 2 )

            def predict( self, X):
                return self.forward(X)


