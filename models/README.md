For start train whole model you need:

* download [archive with datasets](https://yadi.sk/d/asUKcN2e4az0cg) and unpack it to folder datasets.

* download [archive with finetune bert model](https://yadi.sk/d/bZHQ4jjAzyks6g) and unpack it to folder bert.

Now you can run trolldetector.py with flag -m/--mode "fit" for train low-level-classifieres, final troll classifiers and save it trained params.

Also flag -m/--mode can "load" or "predict". If flag "predict", model wait while you print question and answer with "[SEP]" token beetween.
