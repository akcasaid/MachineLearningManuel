import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        '''
        Bu metod, LinearRegression sınıfının başlatıcı (constructor) metodudur.
        İki parametre alır: 
        learning_rate (öğrenme oranı) ve iterations (iterasyon sayısı). 
        Bunlar, modelin öğrenme sürecini kontrol eder.
        self.weights ve self.bias, modelin ağırlıkları ve bias'ı için başlangıçta None olarak atanır. 
        Bu değerler daha sonra fit metodu içinde tanımlanacaktır.
        '''
        self.learning_rate = learning_rate 
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def fit(self,X,y):
        '''
        fit metodu, modelin eğitildiği yerdir ve X (girdi matrisi) ile y (hedef vektör) alır.
        İlk olarak, X matrisinin boyutlarından örnek ve özellik sayısını çıkarır.
        Ağırlıklar ve bias sıfırlanarak başlanır.
        Belirlenen iterasyon sayısı kadar, tahminde bulunulur, 
        hata üzerinden gradyanlar hesaplanır ve öğrenme oranı kullanılarak ağırlıklar ve bias güncellenir.
        
        '''

        num_samples, num_features = X.shape # Örnek sayısı ve öznitelik sayısını al

        #Ağırlıkları ve bias sıfıra ayarla
        self.weights =np.zeros(num_features)
        self.bias = 0

        #Gradient descent
        for _ in range(self.iterations):
            y_pred = self.predict(X) # Mevcut ağırlıkları ve biası kullanarak çıktıyı tahmin et

            #  Gradient hesapla
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1/ num_samples) * np.sum(y_pred - y)


            # Bias ve ağırlık güncelle
            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

    def predict (self, X):
        '''
        Bu metod, verilen X girdi matrisi için tahminler üretir.
        Doğrusal model formülünü (y = wx + b) kullanarak, girdi matrisi X ile ağırlıkların nokta çarpımını alır ve bias'ı ekler.

        '''
        return np.dot(X, self.weights) + self.bias
        


'''
Modelinizin genel yapısı ve fonksiyonları basit bir yapı ile anlatılmıştır, pratikte bazı iyileştirmeler yapılabilir:

* Model başlangıcı: Ağırlıkların ve bias'ın sıfırlanması yerine küçük rassal değerlerle başlamak, bazı durumlarda modelin daha hızlı ve etkili bir şekilde öğrenmesine yardımcı olabilir.
* Veri ölçeklendirme: Modeli eğitmeden önce girdi verilerinin ölçeklendirilmesi (örneğin, standartlaştırma veya min-max normalizasyonu) genellikle model performansını iyileştirir.
* Regularizasyon: Overfitting'i önlemek için L1 veya L2 regularizasyonu gibi teknikler ekleyebilirsiniz.
* Daha fazla durdurma kriteri: Belirli bir hata seviyesinin altına düştüğünde veya ağırlıkların değişimi belirli bir eşiği geçmediğinde iterasyonları durduracak ek koşullar tanımlayabilirsiniz.

'''


# Veri setini oluşturma
np.random.seed(0)
X = 2.5 * np.random.randn(100) + 1.5   # Ortalama 1.5, standart sapma 2.5 olan normal dağılımlı rassal veri
res = 0.5 * np.random.randn(100)       # Hata terimi
y = 2 + 0.3 * X + res                  # Gerçek bağımlı değişken

# Veri setini çizdirme
plt.scatter(X, y)
plt.xlabel("Bağımsız Değişken X (çalışma saatleri)")
plt.ylabel("Bağımlı Değişken y (elde edilen puan)")
plt.title("Oluşturulan Veri Seti")
plt.show()


model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X.reshape(-1, 1), y)

# Eğitilmiş modeli kullanarak tahmin yap
y_pred = model.predict(X.reshape(-1, 1))

# Gerçek değerler ve tahmin edilen değerleri çizdir
plt.scatter(X, y, color='blue', label='Gerçek değerler')
plt.plot(X, y_pred, color='red', label='Tahminler')
plt.xlabel("Bağımsız Değişken X (çalışma saatleri)")
plt.ylabel("Bağımlı Değişken y (elde edilen puan)")
plt.title("Model Tahminleri ve Gerçek Değerler")
plt.legend()
plt.show()


# Tahminleri ve gerçek değerleri yazdır
print("Gerçek Değerler vs. Tahmin Edilen Değerler:")
for i in range(len(y)):
    print(f"{i+1}. Gerçek: {y[i]:.2f}, Tahmin: {y_pred[i]:.2f}")



from sklearn.metrics import mean_squared_error, mean_absolute_error

# MSE, RMSE, ve MAE hesapla
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
