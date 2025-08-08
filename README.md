# created by Ali Arslan

# 🌱 Tarımsal Üretimde Verimlilik Tahmini – Makine Öğrenmesi Projesi

## 🎯 Proje Amacı

Bu projede amaç, tarımsal sensör verileri kullanılarak bir toprağın **verimli (1)** ya da **verimsiz (0)** olduğunu tahmin edebilen bir makine öğrenmesi modeli geliştirmektir. Bu kapsamda veri analizi, ön işleme, model geliştirme, değerlendirme ve yorumlama adımları gerçekleştirilmiştir.

---

## ⚙️ Kurulum ve Gereksinimler

Bu projeyi çalıştırmak için aşağıdaki Python kütüphaneleri gereklidir:

- **Kurulum**

```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imblearn shap
```

- **Gereksinimler**
  Jupyter Notebook ortamında geliştirilmiştir. Google Colab veya Jupyter Notebook üzerinde çalıştırılabilir.

---

## 📁 Veri Kümesi

Kullanılan veri kümesi: `ai-task-2.csv`

### 📌 Değişkenler:

| Değişken Adı      | Açıklama                                   |
| ----------------- | ------------------------------------------ |
| temperature (°C)  | Ortam sıcaklığı                            |
| humidity (%)      | Nem oranı                                  |
| soil_moisture (%) | Toprak nem oranı                           |
| rain (mm)         | Yağış miktarı                              |
| pH                | Toprak pH değeri                           |
| fertilizer_used   | Gübre kullanımı (0 = hayır, 1 = evet)      |
| target            | Hedef değişken (0 = verimsiz, 1 = verimli) |

    dtype:
    temperature        float64
    humidity           float64
    soil_moisture      float64
    rain               float64
    pH                 float64
    fertilizer_used      int64
    target             float64

    - "fertilizier_used" ve "target" değişken verileri kategorikseldir. Diğerleri ise sayısaldır.

---

## 🔍 Veri Analizi

| Özellik                  | Değer                                     |
| ------------------------ | ----------------------------------------- |
| Gözlem Sayısı            | 500                                       |
| Özellik (Feature) Sayısı | 7                                         |
| Null Değerler            | `rain` ve `target` sütunlarında 1'er tane |
| Sıfır Değerler           | `humidity` sütununda 1 adet               |
| Hedef Değişken           | `target` (0 = verimsiz, 1 = verimli)      |

    Eksik ve Anormal Değer Analizi

    🔸 Null Değerler:

        rain: 1 adet eksik değer tespit edildi.
        target: 1 adet eksik değer tespit edildi.

    ✅ Çözüm: 1 adet eksik rain verisi modelleme öncesinde, veri az olmasından dolayı ortalama (mean) ile dolduruldu.

    🔸 Anormal (Sıfır) Değerler:

        humidity: 1 adet değerin 0 olduğu gözlenmiştir. Nem oranının 0 olması çok olası değildir; sensör hatası olabilir.

    ✅ Bu veri kontrol edildi; gerekirse medyan ile değiştirilebilir ama silmeyi tercih ettim.

    🔸 Aykırı (Outlier) Değer Analizi

    ✅ Bu veri setinde IQR (Interquartile Range – Çeyrekler Arası Aralık) yöntemi kullanılmıştır ve aykırı değer tespit edilmemiştir.

---

## 🧹 Veri Ön İşleme

- Null Değer Tespiti ve Çözümü:
  rain ve target değişkenlerine ait 1'er adet null değerli veri tespit edilmiştir.
  rain ortalama ile doldurularak, target ise silinerek çözüme ulaşılmıştır.

- Outlier ve 0 Değer Temizliği:
  humidity == 0 için ilgili veri çıkarılmıştır. Aykıtı (outlier) değer, gerekli yöntem (IQR) kullanılmış ve belirgin bir aykırı değer tespit edilmemiştir.

- Target Veri Tipi Dönüştürme:
  target değişkeninin veri tipi int64 olarak değiştirilmiştir. Çünkü makine öğrenmesi algoritmaları sayısal değer ister. Ayrıca Sklearn, XGBoost ve benzeri kütüphaneler, hedef değişkenin sayısal (numeric) olmasını bekler.

- Hedef Değişken Dengesizliği (Imbalanced Data):
  target sınıf dağılımı (class imbalance) dengesizdir (0 değeri 446, 1 değeri 52 adettir). Bu durum model başarısını olumsuz etkileyebileceği için, sınıf dengesizliğini gidermek için SMOTENC (hem kategorik hem de sayısal sınıflar için ideal) yöntemi uygulanmıştır. Bu sayede hem sayısal hem de kategorik değişkenler (örneğin fertilizer_used) göz önüne alınarak örnekleme (sampling) yapılmıştır. Ayrıca SMOTENC yöntemi sadece eğitim (train) setine uygulanmıştır. Çünkü veri sızıntısı (data leakage) meydana gelebilir.

- Ölçekleme (Scaling):
  StandardScaler ile tüm sayısal özellikler (target ve fertilizer_used harici olanlar) normalize edilmiştir. fertilizer_used zaten ikili (0-1) olduğu için ekstra dönüştürme gerekmez. Model bunu anlayabilir.

- ***

## 🤖 Model Seçimi ve Eğitimi

Aşağıdaki modeller test edilmiştir:

- Logistic Regression
- Random Forest
- XGBoost (Regularized) (En iyi model)
- Support Vector Machine
- Neural Network

**Model Seçimi Kriteri:** Accuracy, Precision, Recall, F1-Score, ROC AUC ve Learning Curve gibi başarı metrikleri uygulanmıştır. Buna göre en iyi model seçilmiştir.

| Metrik             | Açıklama                                                                                                                                                     |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Accuracy**       | Doğru tahminlerin tüm tahminlere oranı                                                                                                                       |
| **Precision**      | Pozitif tahminlerin ne kadarının gerçekten doğru olduğu (yani model verimli toprak olarak tahmin ettiklerinin ne kadarını bilebiliyor)                       |
| **Recall**         | Gerçek pozitiflerin ne kadarının doğru tahmin edildiği (yani model verimli toprakların ne kadarını bilebiliyor)                                              |
| **F1-Score**       | Precision ve Recall'un dengeli harmonik ortalaması                                                                                                           |
| **ROC AUC**        | Sınıflandırmanın genel ayırt etme gücü (yani model verimli verimsiz toprağı ne kadar iyi ayırabiliyor)                                                       |
| **Learning Curve** | Modelin eğitim ve doğrulama verisi üzerindeki başarısının eğitim seti büyüklüğüyle nasıl değiştiği (overfit-underfit olup olmadığını iyi naliz etme metriği) |

- Modeller, hem test verisi performansına hem de 5-Fold Cross Validation sonuçlarına göre karşılaştırılmıştır.
- Tablolarda da görüldüğü üzere, XGBoost ve XGBoost (Regularized) modelleri tüm metriklerde en yüksek performansı sergilemiştir.
- Ek olarak, learning curve grafikleri overfitting veya underfitting gibi problemler olup olmadığını incelemek için kullanılmıştır. XGBoost modeli başta olmak üzere, Random Forest, Neural Networks ve SVM modellerinde de öğrenme eğrisi stabil ve dengeli bir şekilde seyretmiştir. Fakat bu grafik XGBoost'da en iyi şekildedir. Ayrıca Learning curve eğrisine baktığımızda eğitim skoru yüksek, doğrulama skoru da ona yakın, aradaki fark da küçük ve birbirne paralel ise model hem eğitimi öğrenmiş hem de genelleyebiliyor diyebiliriz.

## 📊 Model Performansı ve Adımları

### Modelleme Adımları:

1-Eğitim ve test veri setine ayrım (train-test split)
2-SMOTENC uygulama (sadece train setine)
3-Sayısal değişkenlere standardizasyon (StandardScaler)
4-Model eğitimi (XGBoost, Random Forest vb.)
5-Model değerlendirme (Cross Validation ve Test Set)

### 🔁 K-Fold Cross Validation Sonuçları

Model performansının sadece test verisine bağlı kalmaması için 5-fold çapraz doğrulama uygulanmıştır. Aşağıda her model için ortalama 5-Fold skorları sunulmuştur:

| Model                  | Accuracy | Precision | Recall | F1-Score | ROC AUC |
| ---------------------- | -------- | --------- | ------ | -------- | ------- |
| XGBoost (Regularized)  | 0.9958   | 0.9972    | 0.9944 | 0.9957   | 0.9995  |
| XGBoost                | 0.9958   | 0.9972    | 0.9944 | 0.9957   | 0.9996  |
| Random Forest          | 0.9986   | 1.0000    | 0.9972 | 0.9986   | 1.0000  |
| Neural Network         | 0.9789   | 0.9651    | 0.9944 | 0.9794   | 0.9986  |
| Support Vector Machine | 0.9762   | 0.9648    | 0.9887 | 0.9765   | 0.9980  |
| Logistic Regression    | 0.8582   | 0.8098    | 0.9381 | 0.8688   | 0.8918  |

✅ Çapraz doğrulama sonuçları, modellerin sadece tek bir veri bölünmesine değil, tüm veri setine karşı ne kadar iyi genelleme yapabildiğini ortaya koymaktadır. Burada da en başarılı modeller XGBoost ve Random Forest olmuştur.

## 🧪 Test Verisi Analizi

| Model                      | Accuracy | Precision | Recall | F1-Score | ROC AUC |
| -------------------------- | -------- | --------- | ------ | -------- | ------- |
| **XGBoost (Regularized)**  | 1.00     | 1.0000    | 1.00   | 1.0000   | 1.0000  |
| **XGBoost**                | 1.00     | 1.0000    | 1.00   | 1.0000   | 1.0000  |
| **Random Forest**          | 0.99     | 1.0000    | 0.90   | 0.9474   | 1.0000  |
| **Neural Network**         | 0.96     | 0.8000    | 0.80   | 0.8000   | 0.9811  |
| **Support Vector Machine** | 0.95     | 0.6923    | 0.90   | 0.7826   | 0.9778  |
| **Logistic Regression**    | 0.89     | 0.4737    | 0.90   | 0.6207   | 0.9467  |

- XGBoost ve XGBoost (Regularized) modelleri, test seti üzerinde tüm metriklerde %100 başarı göstermiştir. Bu durum, modelin hem öğrenme kabiliyeti yüksek olduğunu hem de aşırı öğrenme (overfitting) yapmadığını düşündürmektedir.
- Random Forest modeli, %99 doğruluk ve %1 hata ile güçlü bir alternatif olmasına rağmen recall oranı biraz daha düşüktür (0.90). Bu da bazı verimli toprakları verimsiz olarak tahmin etmiş olabileceğini göstermektedir.
- Neural Network ve SVM modelleri, genellikle yüksek başarı sağlamış, özellikle precision değerlerinin düşük olması, verimli toprakların yanlış sınıflandırılma riskini artırmaktadır.
- Logistic Regression, accuracy değeri en düşük model olmuştur (%89 ile). Özellikle precision değerinin düşük olması (0.47), çok fazla false positive (FP) tahmin yapıldığını göstermektedir.

🎯 Genel Değerlendirme:

- XGBoost (Regularized) modeli tüm metriklere bakıldığında en iyi model seçilmiştir.

## 🔗 Korelasyon Analizi

- **Soil Moisture & Rain (0.24)**: Pozitif yönlü korelasyondur. Daha nemli ortamda toprak daha verimli olabilir.
- **Fertilizer Use & Target (0.35)**: Gübre kullanımı verimliliği artırmaktadır ve pozitif yönlüdür.
- **Rain & Target (0.10)**: Yağışın verimliliğe etkisi düşük de olsa pozitif yönlüdür.
- **Temperature & Target (-0.022)**: Sıcaklık ve verimlilik arasında pek de bir anlamlı korelasyon yoktur.
- **Humidity & Target (-0.06)**: Nem ve verimlilik arasında pek de bir anlamlı korelasyon yoktur.
- **Soil Moisture & Rain (0.10)**: Yağış miktarı ve toprak nemi arasında hafif bir pozitif ilişki gözlemlenmiştir, ancak bu ilişki güçlü değildir.

- **Genel Gözlem**: Değişkenler arasında güçlü (> 0.7) korelasyon bulunmamaktadır. Bu durum, değişkenler arasında çoklu doğrusal bağlantı (multicollinearity) probleminin olmadığını göstermektedir.

## 🌿 Değişken Önem Analizi (Feature Importance)

- **Öne Çıkan Bulgular**:
  fertilizer_used: Modelin tahminleri üzerinde en fazla etkiye sahiptir. Gübre kullanımı, toprak verimliliğini güçlü şekilde etkilemektedir.
  soil_moisture: Toprak nem oranı, model kararlarında ikinci en önemli faktördür. Nemli toprakların daha verimli olması beklenmektedir.
  pH: Toprak pH değeri, verimliliğe önemli katkı sağlayan değişkenlerden biridir.
  rain: Yağış miktarı belirli düzeyde etkilidir ancak katkısı daha sınırlıdır.
  humidity ve temperature: Model çıktısı üzerindeki etkileri oldukça düşüktür. Bu değişkenler, verimlilik tahmininde model tarafından daha az dikkate alınmaktadır.

## 📚 Kullanılan Kaynaklar / Framework'ler

- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
- [imblearn](https://imbalanced-learn.org/stable/)

---

## 📝 Ek Açıklamalar
