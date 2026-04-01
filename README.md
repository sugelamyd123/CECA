<<<<<<< HEAD

## Introduction
PyTorch code for *Tackling Alignment Ambiguity in Person Retrieval through Conversational Attribute Mining*
⚠️⚠️⚠️ 20241128 - Fixing [issue 8](https://github.com/QinYang79/RDE/issues/8) will cause some performance degradation in noisy scenes. You can comment out "line217 model.train()" in processor/processor.py. Thanks to *Xiangwen Deng* for pointing this out.

### News!

- We release the training code and training logs.
- Accepted by CVPR 2026.





## Requirements and Datasets
- Same as [IRRA](https://github.com/anosorae/IRRA)



## Training and Evaluation

### Training new models

```
sh run_rde.sh
```

### Evaluation
Modify the  ```sub``` in the ```test.py``` file and run it.
```
python test.py
```

 




## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## Acknowledgements
The code is based on [IRRA](https://github.com/anosorae/IRRA) licensed under Apache 2.0.
=======
