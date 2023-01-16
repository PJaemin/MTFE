# MTFE

### Jaemin Park, An Gia Vien, Minhee Cha, Hanul Kim, and Chul Lee
Official pytorch implementation for **"Multiple Transformation Function Estimation for Image Enhancement"**

<p float="left">
  &emsp;&emsp; <img src="overview.PNG" width="800" />
</p>

## Preparation
### Training data: [Download from GoogleDrive](https://drive.google.com/file/d/1jekxUtXmcU79DfnyTMbLEUm9y6vQwuVU/view?usp=sharing)
The ZIP file contains three test datasets:
- LOL dataset: 485 image pairs
- FiveK dataset: 4,500 image pairs
- EUVP dataset: 11,435 image pairs

### Testing samples: [Download from GoogleDrive](https://drive.google.com/file/d/1bnmfDTkcK-Sq2KGIWnv9QmEZUWyHg4x5/view?usp=sharing)
The ZIP file contains three test datasets:
- LOL dataset: 15 image pairs
- FiveK dataset: 500 image pairs
- EUVP dataset: 515 image pairs

### Pretrained weights: [Download from GoogleDrive](https://drive.google.com/file/d/1SM54xIQ5q-vtdPdg-0LVlGjsM98YqWhR/view?usp=sharing)
The ZIP file contains weight files trained with each training dataset.

## Training
Run below commend:
```
python lowlight_train.py
```

## Testing
Run below commend:
```
python lowlight_test.py
```

## Citation
If you find this work useful for your research, please consider citing our paper:
```
@article{Park2023,
    author={{Park, Jaemin and Vien, An Gia and Cha, Minhee and Kim, Hanul and Lee, Chul}},
    booktitle={Journal of Visual Communication and Image Representation},
    title={Multiple Transformation Function Estimation for Image Enhancement}, 
    year={2023},
    volume={},
    number={},
    pages={},
    doi={}}
}
```

## License
See [MIT License](https://github.com/PJaemin/MTFE/blob/main/LICENSE)


