# Sentiment Analysis on Financial Documents
Dataset from  https://github.com/nlp-chula/finnlp-sentiment

In this project, a sentiment analysis model was developed for Form 56-1 documents of 50 companies between the years 2015 and 2019. The analysis was conducted using NLP bag-of-words method.


### Dataset
![image](https://github.com/Dolphuwadol/sentiment-analysis-on-fin-doc/assets/121854744/d9d6bfde-d06c-43f6-9d5f-ffac5e65c564)


### Data Dictionary
Sentiment ที่มีต่อเอกสารใน 3 ส่วน ได้แก่ 
- 1.ส่วนการบริหารจัดการความเสี่ยง 
- 2.ส่วนการวิเคราะห์และคำอธิบายของฝ่ายจัดการ 
- 3.ส่วนการขับเคลื่อนธุรกิจเพื่อความยั่งยืน

| ประเภทขั้วอารมณ์ (Sentiment) |                                                                                                                            คำนิยาม/คำอธิบาย                                                                                                                           |
|:-------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Negative                  | ความรู้สึกที่เป็นลบ โดยยึดโยงจากมุมมองนักลงทุนทั่วไป หรือนักวิเคราะห์หลักทรัพย์/การเงิน - ข้อความสื่อถึงการเปลี่ยนแปลง หรือ ผลกระทบที่เป็น แง่ลบ ต่อประเภทของทัศนคติ (Aspect)  - ข้อความที่เป็น ลบ เป็นได้ทั้ง ข้อเท็จจริง ความเห็น ความคิด อารมณ์ หรือ การตัดสินใจ เกี่ยวกับบริษัท                                         |
| Neutral                   | ความรู้สึกที่เป็นกลาง โดยยึดโยงจากมุมมองนักลงทุนทั่วไป หรือนักวิเคราะห์หลักทรัพย์/การเงิน - ข้อความ ไม่ได้สื่อ ถึงการเปลี่ยนแปลง หรือ ผลกระทบที่เป็นด้านใดด้านหนึ่ง ต่อประเภทของทัศนคติ - ข้อความที่อาจจะเป็น กลาง เป็นเพียงคำกล่าวของข้อเท็จจริงเกี่ยวกับบริษัท - ความเห็น ความคิด อารมณ์ ที่แสดงนั้นเป็นไปอย่างกลาง ๆ (moderate) |
| Positive                  | ความรู้สึกที่เป็นบวก โดยยึดโยงจากมุมมองนักลงทุนทั่วไป หรือนักวิเคราะห์หลักทรัพย์/การเงิน - ข้อความสื่อถึงการเปลี่ยนแปลง หรือ ผลกระทบที่เป็น แง่บวก ต่อประเภทของทัศนคติ (Aspect) - ข้อความที่เป็น บวก เป็นได้ทั้ง ข้อเท็จจริง ความเห็น ความคิด อารมณ์ หรือ การตัดสินใจ เกี่ยวกับบริษัท         

| ประเภททัศนคติ (Aspects) |                                                                                                                                                                                       คำนิยาม/คำอธิบาย                                                                                                                                                                                      |
|:---------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Brand                 | ภาพลักษณ์บริษัท/ตราสินค้า (แบรนด์) - แบรนด์ รวมถึง ภาพลักษณ์ตราสินค้า ภาพลักษณ์องค์กร ภาพลักษณ์สินค้า - การตลาด รวมถึง การโฆษณา PR รางวัลการตลาด การส่งเสริมการขาย การทำสปอนเซอร์                                                                                                                                                                                                                                  |
| Product/Service       | ผลิตภัณฑ์หรือบริการของบริษัท - การประกาศ ออกสินค้า/บริการใหม่ การทดลองตลาดสินค้า/บริการ - การเปลี่ยนแปลง อัพเกรด/ดาวน์เกรด เรียกคืน อนุมัติ - ความร่วมมือกับบริษัทอื่น เช่น การทำ licensing พันธมิตร (alliance) หุ้นส่วนธุรกิจ (partnership) การทำ MOU, Joint Venture                                                                                                                                                           |
| Environment           | การดำเนินงานด้านสิ่งแวดล้อม - นโยบายด้านสิ่งแวดล้อม นิเวศวิทยา ภาวะโลกร้อน (Global Warming) การเปลี่ยนแปลงสภาพภูมิอากาศ (Climate Change) - การสร้างของเสีย การปล่อยมลพิษ - กิจกรรม CSR (Corporate Social Responsibility) ต่าง ๆ ทั้งในด้านสิ่งแวดล้อม                                                                                                                                                                 |
| Social & People       | สังคม และผู้คน - การจ้างงาน - การเปลี่ยนแปลงในการจ้างงานของพนักงานบริษัท (การจ้าง หรือ เลิกจ้าง) - ค่าใช้จ่ายในการจ้างงาน (compensation) - การเปลี่ยนแปลงผู้บริหาร (เช่น CEO, ผู้บริหารระดับสูง) - การหยุดงาน (strike) - ประเด็นอื่น ๆ ที่เกี่ยวกับการจ้างงาน  กิจกรรม CSR ต่าง ๆ ทั้งในด้านสังคม ที่เกี่ยวกับพนักงาน แรงงาน ลูกค้า ชุมชน ท้องถิ่น หรือผู้มีส่วนได้ส่วนเสียอื่น ๆ                                                                            |
| Governance            | ธรรมาภิบาลของบริษัท - การเปลี่ยนแปลงของคณะกรรมการบริษัท (Board of Directors) - นโยบายการกำกับดูแลบริษัท และบริษัทย่อย - ความโปร่งใสในการดำเนินงาน จริยธรรม การตรวจสอบของผู้บริหาร                                                                                                                                                                                                                             |
| Economics             | การบรรยายถึงเศรษฐกิจมหภาค ที่อาจส่งผลต่อบริษัท - สภาวะเศรษฐกิจของประเทศและโลก นโยบายเศรษฐกิจต่าง ๆ นโยบายการค้าขายระหว่างประเทศ เช่น FTA (Free trade agreements) -ดัชนีทางเศรษฐกิจต่าง ๆ เช่น GDP อัตราดอกเบี้ย อัตราเงินเฟ้อ อัตราการว่างงาน รายได้ประชาชาติ อัตราแลกเปลี่ยนค่าเงิน - แนวโน้มเศรษฐกิจใน อุตสาหกรรม ประเทศ และโลก                                                                                              |
| Political             | การเมือง - การเปลี่ยนแปลงทางการเมือง เช่น การเลือกตั้ง การทำรัฐประหาร การเคลื่อนไหวทางการเมือง ความไม่สงบทางการเมือง สงคราม - นโยบายของภาครัฐ - นโยบายภาษี                                                                                                                                                                                                                                                |
| Legal                 | ข้อพิพาททางกฎหมาย หรือการตัดสินใจที่เกี่ยวข้องกับกฎหมาย รวมถึง การสอบสวน การกล่าวหา การฟ้องร้อง คดีความ การถูกดำเนินคดี การฉ้อโกง การฟอกเงิน การยอมความ การจ่ายค่าเสียหาย คำพิพากษา กฎหมาย และประเด็นทางกฎหมายอื่นๆ                                                                                                                                                                                                   |
| Dividend              | เงินปันผล คือ เงินจ่ายให้แก่ผู้ถือหุ้นของบริษัท - การจ่ายเงินปันผล อาจมาในรูปของ เงินสด หุ้น หรือสินทรัพย์รูปแบบอื่น - สังเกตการเปลี่ยนแปลงที่เกี่ยวกับเงินปันผลในด้าน การคาดการณ์ (forecast) การรายงาน การประกาศจ่าย                                                                                                                                                                                                              |
| Investment            | การลงทุน - เงินลงทุน (capital expenditure) ในตัวบริษัท บริษัทย่อยหรือร่วม สาขา การลงทุนในโครงสร้างการผลิต (เช่น โรงงาน) การลงทุนในสินค้าหรือบริการ - การลงทุนในการวิจัยและพัฒนา (Research & Development) - เหตุการณ์ที่เกี่ยวกับ โรงงาน ตึกสำนักงาน อาคาร ร้านค้า สาขา โกดัง หรืออสังหาริมทรัพย์อื่น ๆ - ยกเว้น การควบรวมกิจการ (M&A)                                                                                                  |
| M&A                   | การควบรวมกิจการของบริษัท (Merger and Acquisition) - Merger คือ การที่บริษัทตั้งแต่ 2 บริษัทขึ้นไปทำการควบรวมกิจการเข้าด้วยกันแล้วเกิดเป็นบริษัทใหม่ - Acquisition คือ การที่บริษัทหนึ่ง เข้าไปซื้อกิจการบางส่วนหรือทั้งหมด ของอีกบริษัทหนึ่ง ซึ่งเราสามารถแบ่งออกได้เป็น 2 กรณีด้วยกัน  * Share Acquisition คือ การที่ผู้ซื้อเข้ามาซื้อหุ้นของบริษัทบางส่วน หรือทั้งหมด  * Asset / Business Acquisition คือ การที่ผู้ซื้อเข้ามาซื้อทรัพย์สิน, หน่วยธุรกิจบางส่วนหรือทั้งหมด ของกิจการ |
| Profit/Loss           | ผลประกอบการบริษัท - นับรวมไปถึง รายได้ (Revenue) ยอดขาย (Sales) ต้นทุนขาย (Costs of Goods Sold) ค่าใช้จ่ายต่าง ๆ (Expenses) - ตัวเลขทางการเงิน (Financials) หรืออัตราส่วนทางการเงิน (Financial Ratios) ต่าง ๆ - กำไร (หรือ ขาดทุน) สุทธิ คือ รายได้หลังหักค่าใช้จ่ายทั้งหมด - กำไรสุทธิ = รายได้ - ต้นทุนขาย - ค่าใช้จ่ายในการขายและบริหาร - ค่าใช้จ่ายดอกเบี้ย - ภาษี - รวมถึงการเปลี่ยนแปลงราคาหุ้นในตลาดหลักทรัพย์                           |
| Rating                | อันดับความน่าเชื่อถือของบริษัท - การจัดเรตติ้ง การจัดอันดับความน่าเชื่อถือของตัวองค์กร หรือ การจัดอันดับความน่าเชื่อถือของตราสารหนี้แต่ละตัว ที่จะสะท้อนความสามารถในการชำระหนี้ของผู้ออกตราสาร - สถาบันจัดอันดับความน่าเชื่อถือในไทยมี 2 แห่งคือ บจก.ทริสเรทติ้ง (TRIS) และ บจก. ฟิทช์ เรทติ้งส์ (Fitch) - ข้อเสนอแนะ หรือ คำแนะนำของนักวิเคราะห์ (เช่น คำแนะนำ ซื้อ/ขาย/ถือ) เกี่ยวกับ รวมถึงการเปลี่ยนแปลงคำแนะนำด้วย                                             |
| Financing             | การกู้ยืมเงิน (loan) การทำ syndicated loan การออกหุ้นกู้ (bond) การเพิ่มทุนในตลาดหลักทรัพย์ การซื้อหุ้นกลับคืน (stock repurchase) การให้กู้ยืมระหว่างกันบริษัทที่เกียวข้อง/บริษัทลูก การทำ IPO (initial public offering) การทำ private placement หุ้น การทำ tender offer การเพิ่ม/ลดทุน จาก VC, angel investor                                                                                                                    |
| Technology            | การเปลี่ยนแปลงด้านเทคโนโลยี สารสนเทศ การใช้ automation การใช้ AI นวัตกรรมต่าง ๆ การเข้าถึง (Access), licensing, patent และ ทรัพย์สินทางปัญญาเทคโนโลยี                                                                                                                                                                                                                                                    |
| Others                | หัวข้ออื่น ๆ หัวข้อการเปลี่ยนแปลงในด้านอื่น ๆ นอกเหนือจากที่กล่าวข้างต้น อาทิ เช่น การเปลี่ยนแปลงเทคโนโลยี ภัยพิบัติ โรคระบาด                                                                                                                                                                                                                                                                                       |
### Visualization
![newplot](https://github.com/Dolphuwadol/sentiment-analysis-on-fin-doc/assets/121854744/7c6a0a19-9385-46a5-b34e-d002c8448060)
![newplot (1)](https://github.com/Dolphuwadol/sentiment-analysis-on-fin-doc/assets/121854744/1937612b-3625-4ee6-8233-a437a60433ae)

### Output
Sentiment Analysis & Aspect
```
sample = ['โรงงานโออิชิฟู้ด เซอร์วิส ได้ดำเนินการปรับปรุงประสิทธิภาพกระบวนการผลิตปลาซาบะโดยนำระบบสายพานทดแทนการผลิตแบบเดิมสามารถลดการใช้วัตถุดิบลงได้ 10,000 กิโลกรัมต่อปี',
          'การให้ความช่วยเหลือลูกค้าของธนาคารที่ได้รับผลกระทบจากสถานการณ์การแพร่ระบาดของโรคติดเชื้อไวรัสโคโรนา 2019',
          'ธนาคารไทยพาณิชย์มีการปรับตัวและพัฒนาด้านเทคโนโลยี เพื่อตอบสนองต่อไลฟ์สไตล์ผู้บริโภคและภาคธุรกิจที่เปลี่ยนไปอย่างพอเพียงและรวดเร็ว',
          'ธนาคารให้ความสำคัญอย่างยิ่งต่อการพัฒนาระบบความปลอดภัยสารสนเทศ ซึ่งรวมถึงการเคารพและรักษาสิทธิของข้อมูล ความเป็นส่วนตัวของลูกค้า']

array(['Positive', 'Neutral', 'Neutral', 'Neutral'], dtype=object)  
array([[0.0707893 , 0.35536414, 0.57384656],
       [0.09622716, 0.65160287, 0.25216997],
       [0.11071075, 0.5565413 , 0.33274795],
       [0.02162044, 0.80186629, 0.17651327]])

# Aspect
array(['Environment', 'Social&People', 'Social&People', 'Social&People'],
      dtype=object)

array([[0.00438823, 0.00264637, 0.04195515, 0.46690401, 0.0168527 ,
        0.02036079, 0.01698312, 0.00625141, 0.00319269, 0.05966361,
        0.01405899, 0.2339503 , 0.0361405 , 0.00148008, 0.05993874,
        0.01523332],
       [0.02043458, 0.00533704, 0.09309781, 0.02754014, 0.02809226,
        0.09907379, 0.01204472, 0.01426337, 0.0057129 , 0.20318476,
        0.01191806, 0.08489957, 0.13344138, 0.00223627, 0.24662362,
        0.01209974],
       [0.01997702, 0.00369334, 0.09782183, 0.03289479, 0.01297738,
        0.06952814, 0.01525373, 0.00843261, 0.00488289, 0.12804574,
        0.01311572, 0.12591614, 0.04928716, 0.00186637, 0.39141903,
        0.02488811],
       [0.02099831, 0.0046979 , 0.0219506 , 0.02593044, 0.01649465,
        0.05988751, 0.01016675, 0.01155934, 0.00426674, 0.19924421,
        0.00721741, 0.13506875, 0.0258411 , 0.00166074, 0.37210472,
        0.08291085]])
```
