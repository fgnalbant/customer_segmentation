import datetime as dt
import pandas as pd
import seaborn as sns


###GÖREV 1###

pd.set_option("display.max_columns", None)  # Tüm sütunlar görünsün
pd.set_option("display.float_format", lambda x: "%.3f" % x)

#ADIM 1 VERİSETİNİ OKUMA
df_ = pd.read_csv("<path>/<customer_dbinfo>.csv")
df = df_.copy()

#ADIM 2
df.head(10)
df.info()
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df.dtypes
df.shape

#Adım 3: Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
#alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
df["TotalOrderNumber"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_ever"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df.head()

#Adım 4: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.dtypes

for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])

#Adım 5: Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.
df["master_id"].nunique()
df["order_channel"].value_counts()
df.groupby("order_channel").agg({"TotalOrderNumber" : "sum",
                                 "customer_value_total_ever": "sum"})

#Adım 6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df.groupby("master_id").agg({"customer_value_total_ever": "sum"}).sort_values("customer_value_total_ever", ascending=False).head(10)

#Adım 7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.groupby("master_id").agg({"TotalOrderNumber": "sum"}).sort_values("TotalOrderNumber", ascending=False).head(10)

#Adım 8: Veri ön hazırlık sürecini fonksiyonlaştırınız
def prepare_data(dataframe):
    dataframe["TotalOrderNumber"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total_ever"] = (dataframe["order_num_total_ever_online"] * dataframe["customer_value_total_ever_online"]) + \
                                             (dataframe["order_num_total_ever_offline"] * dataframe["customer_value_total_ever_offline"])

    for col in dataframe.columns:
        if "date" in col:
            dataframe[col] = pd.to_datetime(dataframe[col])

    return dataframe.head()

prepare_data(df)
df.info()


###GÖREV 2###

#Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız,

#Recency: Müşterinin ne kadar yeni olduğunu ifade eder.
#Frequency : Müşterinin alışveriş sıklığını ifade eder.
#Monetary : Müşterinin yaptığı alışverişle firmaya bıraktığı parasal değeri ifade eder.

#Adım 2 Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız
#Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız

df.head()
df["last_order_date"].max()

today_date = dt.datetime(2021, 6, 1)

rfm = df.groupby("master_id").agg({"last_order_date": lambda last_order_date: (today_date - last_order_date.max()).days,
                                     "TotalOrderNumber": lambda TotalOrderNumber: TotalOrderNumber,
                                     "customer_value_total_ever": lambda customer_value_total_ever: customer_value_total_ever.sum()})

#Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.

rfm.columns = ["recency", "frequency", "monetary"]
rfm.head()

### Görev 3: RF Skorunun Hesaplanması ###

# Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
#Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm.head()

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

#Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.

rfm["RF_SCORE"] = (rfm["recency_score"].astype(str)) + (rfm["frequency_score"].astype(str))
rfm.head()

### Görev 4: RF Skorunun Segment Olarak Tanımlanması ###

#Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız
#Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz

seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_Risk",
    r"[1-2]5": "cant_loose",
    r"3[1-2]": "about_to_sleep",
    r"33": "need_attention",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalist",
    r"5[4-5]": "champions"
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
rfm.head()


### Görev 5: Aksiyon Zamanı ###x

# Adım 1: Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean"])

#Adım 2: RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv olarak kaydediniz

#CASE1 : a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
#       tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
#       iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
#       yapan kişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına kaydediniz

rfm=pd.merge(rfm, df, how='left', on='master_id')
womendf = rfm[["master_id", "segment", "interested_in_categories_12"]]


womendf = womendf.loc[(womendf["interested_in_categories_12"].str.contains("KADIN")) &
                      ((womendf["segment"] == "loyal_customers") | (womendf["segment"] == "champions"))]

womendf[["master_id"]].to_csv("a_target_customer_id.csv")


#CASE2 : b. Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
#           iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni
#           gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.

boys_40df = rfm[["master_id", "segment", "interested_in_categories_12"]]
boys_40df


boys_40df = boys_40df.loc[((boys_40df["interested_in_categories_12"].str.contains("COCUK")) |
                           (boys_40df["interested_in_categories_12"].str.contains("ERKEK"))) &
                           ((boys_40df["segment"] == "hibernating") |
                           (boys_40df["segment"] == "cant_loose") |
                           (boys_40df["segment"] == "new_customers"))]