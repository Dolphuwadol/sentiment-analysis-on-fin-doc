# Sentiment Analysis on Financial Documents
Dataset from  https://github.com/nlp-chula/finnlp-sentiment


![newplot](https://github.com/Dolphuwadol/sentiment-analysis-on-fin-doc/assets/121854744/7c6a0a19-9385-46a5-b34e-d002c8448060)
![newplot (1)](https://github.com/Dolphuwadol/sentiment-analysis-on-fin-doc/assets/121854744/1937612b-3625-4ee6-8233-a437a60433ae)

### Output
```
sample = ['โรงงานโออิชิฟู้ด เซอร์วิส ได้ดำเนินการปรับปรุงประสิทธิภาพกระบวนการผลิตปลาซาบะโดยนำระบบสายพานทดแทนการผลิตแบบเดิมสามารถลดการใช้วัตถุดิบลงได้ 10,000 กิโลกรัมต่อปี',
          'การให้ความช่วยเหลือลูกค้าของธนาคารที่ได้รับผลกระทบจากสถานการณ์การแพร่ระบาดของโรคติดเชื้อไวรัสโคโรนา 2019',
          'ธนาคารไทยพาณิชย์มีการปรับตัวและพัฒนาด้านเทคโนโลยี เพื่อตอบสนองต่อไลฟ์สไตล์ผู้บริโภคและภาคธุรกิจที่เปลี่ยนไปอย่างพอเพียงและรวดเร็ว',
          'ธนาคารให้ความสำคัญอย่างยิ่งต่อการพัฒนาระบบความปลอดภัยสารสนเทศ ซึ่งรวมถึงการเคารพและรักษาสิทธิของข้อมูล ความเป็นส่วนตัวของลูกค้า']
sample_feature = tfidf_fit.transform(sample)
model.predict(sample_feature)
     
array(['Positive', 'Neutral', 'Neutral', 'Neutral'], dtype=object)

model_proba = model.predict_proba(sample_feature)
model_proba
     
array([[0.0707893 , 0.35536414, 0.57384656],
       [0.09622716, 0.65160287, 0.25216997],
       [0.11071075, 0.5565413 , 0.33274795],
       [0.02162044, 0.80186629, 0.17651327]])
```
