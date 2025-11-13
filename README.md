### تحلیل داده‌های سری زمانی جهت پیش‌‌بینی موقعیت هواپیما با استفاده از شبکه‌هاي ترنسفورمر گرافی و بیزی

پیش‌بینی موقعیت آینده هواپیماها یکی از مسائل کلیدی در مدیریت ترافیک هوایی است و نقش مهمی در ارتقای ایمنی، بهینه‌سازی مصرف سوخت و کاهش تأخیرهای پروازی ایفا می‌کند. با وجود پیشرفت‌های اخیر در کاربرد مدل‌های یادگیری عمیق، بسیاری از مدل‌های موجود قادر به تحلیل تعاملات پیچیده میان هواپیماها و تخمین عدم قطعیت پیش‌بینی‌ها نیستند. در این پژوهش، یک چارچوب ترکیبی مبتنی بر شبکه‌های عصبی ترنسفورمر، BiLSTM، گرافی و شبکه‌های عصبی ارائه شده است. این مدل علاوه بر پیش‌بینی موقعیت‌های آتی هواپیما با استفاده از داده‌های موقعیتی گذشته و تاثیر ترافیک سایر هواپیماها، امکان برآورد عدم قطعیت پیش‌بینی‌ها را نیز فراهم می‌کند. ابرپارامترهای مدل با استفاده از الگوریتم فراابتکاری بهینه‌سازی ازدحام ذرات تنظیم شد. مدل پیشنهادی بر روی داده‌های ADS-B مجموعه داده TartanAviation آموزش داده شد و نتایج نشان داد که نسبت به مدل پایه بهبود هفت درصدی در معیار متوسط خطای جابه‌جایی (ADE) و چهار درصدی در معیار خطای جابه‌جایی نهایی (FDE) داشته است. علاوه بر این، زمان آموزش مدل 11 درصد کاهش یافت. تحلیل انحراف معیار خروجی‌های رمزگشای بیزی نشان داد که عدم قطعیت به طور منطقی با افزایش افق پیش‌بینی افزایش می‌یابد. در مجموع، ترکیب مزایای شبکه‌های عصبی گرافی، ترنسفورمر، BiLSTM و شبکه‌های عصبی بیزی چارچوبی کارآمد برای تحلیل داده‌های مکانی-زمانی در پیش‌بینی مسیر پرواز فراهم می‌کند. برای ارزیابی تعمیم‌پذیری، داده‌های پروازهای فرودگاه بین‌المللی امام خمینی گردآوری و برای آموزش و ارزیابی مجدد مدل مورد استفاده قرار گرفت.





### Environment

```bash
pip install numpy==1.18.1
pip install torch==1.7.0
pip install pyyaml==5.3.1
pip install tqdm==4.45.0
```

### Data
The data source used are ASDE-X data from Sherlock. Here, we anonymized the data by,

- Remove the real-world unix timestamp and replace them by absolute time steps (e.g. 5, 10, 15, 20).
- The flight callsign are masked with a unique agent id (integer).

For this experiment, only four columns of ASDE-X needed. They are time, id, latitude, and longitude.

The code used for processing and anonymized the raw ASDE-X data can be found in Part 1 Data Processing Demo. Data are saved in ```./data/iff/atl/2019080x/true_pos_.csv```


### To Train an Example
This command is to train model using the ASDE-X data from Aug 1st, 2019 to Aug 6th, 2019, and test the trained model with data from Aug 7th, 2019.

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

Code of Part 2 borrowed heavily from [here](https://github.com/Majiker/STAR)
