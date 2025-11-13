### تحلیل داده‌های سری زمانی جهت پیش‌‌بینی موقعیت هواپیما با استفاده از شبکه‌هاي ترنسفورمر گرافی و بیزی

پیش‌بینی موقعیت آینده هواپیماها یکی از مسائل کلیدی در مدیریت ترافیک هوایی است و نقش مهمی در ارتقای ایمنی، بهینه‌سازی مصرف سوخت و کاهش تأخیرهای پروازی ایفا می‌کند. با وجود پیشرفت‌های اخیر در کاربرد مدل‌های یادگیری عمیق، بسیاری از مدل‌های موجود قادر به تحلیل تعاملات پیچیده میان هواپیماها و تخمین عدم قطعیت پیش‌بینی‌ها نیستند. در این پژوهش، یک چارچوب ترکیبی مبتنی بر شبکه‌های عصبی ترنسفورمر، BiLSTM، گرافی و شبکه‌های عصبی ارائه شده است. این مدل علاوه بر پیش‌بینی موقعیت‌های آتی هواپیما با استفاده از داده‌های موقعیتی گذشته و تاثیر ترافیک سایر هواپیماها، امکان برآورد عدم قطعیت پیش‌بینی‌ها را نیز فراهم می‌کند. ابرپارامترهای مدل با استفاده از الگوریتم فراابتکاری بهینه‌سازی ازدحام ذرات تنظیم شد. مدل پیشنهادی بر روی داده‌های ADS-B مجموعه داده TartanAviation آموزش داده شد و نتایج نشان داد که نسبت به مدل پایه بهبود هفت درصدی در معیار متوسط خطای جابه‌جایی (ADE) و چهار درصدی در معیار خطای جابه‌جایی نهایی (FDE) داشته است. علاوه بر این، زمان آموزش مدل 11 درصد کاهش یافت. تحلیل انحراف معیار خروجی‌های رمزگشای بیزی نشان داد که عدم قطعیت به طور منطقی با افزایش افق پیش‌بینی افزایش می‌یابد. در مجموع، ترکیب مزایای شبکه‌های عصبی گرافی، ترنسفورمر، BiLSTM و شبکه‌های عصبی بیزی چارچوبی کارآمد برای تحلیل داده‌های مکانی-زمانی در پیش‌بینی مسیر پرواز فراهم می‌کند. برای ارزیابی تعمیم‌پذیری، داده‌های پروازهای فرودگاه بین‌المللی امام خمینی گردآوری و برای آموزش و ارزیابی مجدد مدل مورد استفاده قرار گرفت.





### آماده سازی محیط توسعه

```bash
pip install numpy==1.18.1
pip install torch==1.7.0
pip install pyyaml==5.3.1
pip install tqdm==4.45.0
```

### دیتاست
برای آموزش و تست از دیتاست TartanAviation به‌عنوان یک مجموعه داده چندوجهی و جامع در حوزه عملیات هوانوردی در نواحی ترمینالی فرودگاه‌ها استفاده شده است. این مجموعه داده توسط دانشگاه کارنگی ملون (Carnegie Mellon University) و در قالب پروژه‌ای از آزمایشگاه Airlab  وابسته به دانشکده علوم کامپیوتر این دانشگاه توسعه یافته است. هدف از طراحی این مجموعه، فراهم‌سازی بستری برای پژوهش‌های پیشرفته در زمینه ادراک چندوجهی، تحلیل ایمنی هوانوردی، و توسعه سیستم‌های هوشمند پشتیبانی تصمیم‌گیری در فرودگاه‌ها بوده است. دیتاست TartanAviation  منبعی ارزشمند از داده‌های هم‌زمان شامل تصاویر ویدیویی، مکالمات صوتی کنترل ترافیک هوایی، و داده‌های موقعیت‌یابی ADS-B است که در بازه‌های زمانی متنوع و در دو فرودگاه متفاوت در ایالات متحده (یکی دارای برج مراقبت و دیگری بدون برج) گردآوری شده‌اند. این مجموعه شامل بیش از ۳.۱ میلیون تصویر، ۳۳۷۴ ساعت گفت‌وگوی صوتی میان خلبانان و کنترلرها، و ۶۶۱ روز داده‌های موقعیت‌یابی پروازهاست. تمامی داده‌ها پس از جمع‌آوری، پالایش، پردازش و اعتبارسنجی شده‌اند و همراه با کد منبع مراحل جمع‌آوری و پیش‌پردازش، به‌صورت متن‌باز منتشر شده‌اند. ویژگی چندوجهی و تنوع شرایط جوی و عملیاتی این مجموعه را به منبعی منحصربه‌فرد برای پژوهش‌های مرتبط با تحلیل سری‌های زمانی، پردازش تصویر، درک زبان طبیعی، و مدل‌های یادگیری چندوجهی در زمینه هوانوردی تبدیل کرده است. در این تحقیق، از زیرمجموعه‌ی مربوط به داده‌های ADS-B پروازها برای آموزش و ارزیابی مدل پیش‌بینی  موقعیت آتی هواپیماها استفاده شده است  با توجه به منابع پردازشی محدود در این پژوهش از گام‌های زمانی 5 ثانیه‌ای استفاده شده است. 


### آموزش و تست
برای آموزش و تست مدل از دستور زیر استفاده میشود 

```
python trainval.py --num_epochs 300 --start_test 250 --neighbor_thred 10 --skip 5 --seq_length 20 --obs_length 12 --pred_length 8 --randomRotate False --learning_rate 0.0015 --sample_num 20
```

The model will be trained for 300 epochs, and the testing start at epoch 250, with a learning rate of 0.0015. In the test phase, the trained model will be sampled 20 times.

And with the following paramaters, 
- The neighboring aircraft threshold is 10km (~3nm). 
- The timestamp in the processed flight data is 5 seconds. 
- The total length of the sequence is 20 timestamps, where observation is 12 timestamps, prediction is 8 timestamps.

During training, the trained model with a new best FDE on the test dataset will be saved in the output folder.


### Source Code
In ```./src```, there are multiple Python scripts,

- ```utils.py```: Data pre-processing before training. For instance, using the previous command, there will be 24 training batches and 7 testing batches.
- ```lrt_linear.py```: Bayesian Linear Layer used to build the decoder using local reparameterization trick.
- ```multi_attention_forward.py```: The multi-head attention layer.
- ```bstar.py```: The code to build the B-STAR architecture.
- ```processor.py```: Training and testing function.


### Reference

Code borrowed heavily from [here](https://github.com/ymlasu/para-atm-collection/tree/master/air-traffic-prediction/MultiAircraftTP)
