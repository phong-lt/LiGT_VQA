# LiGT: Layout-infused Generative Transformer for Visual Question Answering on Vietnamese Receipts

This repository includes the ReceiptVQA dataset and the Pytorch implementation of the LiGT method and other evaluated baselines. For more details, please check out our paper or [preprint](https://arxiv.org/abs/2502.19202).

## Dataset

The **ReceiptVQA** dataset encompasses **60,000+** Vietnamese question-answer pairs manually annotated from **9,000+** receipt images. [ReceiptVQA](https://drive.google.com/drive/folders/1mWTKRWm0FDmw1I200ixIjFzg3ij7wcy-) is publicly available exclusively for research purposes.

## Method

> LiGT (Layout-infused Generative Transformer), a layout-aware encoder-decoder architecture designed to leverage embedding layers of language models to operate layout embeddings, minimizing the use of additional neural modules.

## Script

### Setup

1. Clone the repository:
    ```
    git clone https://github.com/phong-lt/LiGT_VQA
    ```
2. Install the required packages:
    ```
    pip install -r /LiGT_VQA/requirements.txt
    ```

### Usage

To run the main script:
```bash
python LiGT_VQA/run.py \
	# config file path
	--config-file LiGT_VQA/config/ligt.yaml \
 
	# mode: train - to train models, eval - to evaluate models, predict - to predict trained models
	--mode train \

	# evaltype: last - evaluate lattest saved model, best - evaluate best-score saved model 
	--evaltype last \
	
	# predicttype: last - predict lattest saved model, best - predict best-score saved model 
	--predicttype best \
```

# Citation

If you use LiGT or the ReceiptVQA dataset, please cite the following paper:

```
@misc{le2025ligtlayoutinfusedgenerativetransformer,
      title={LiGT: Layout-infused Generative Transformer for Visual Question Answering on Vietnamese Receipts}, 
      author={Thanh-Phong Le and Trung Le Chi Phan and Nghia Hieu Nguyen and Kiet Van Nguyen},
      year={2025},
      eprint={2502.19202},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.19202}, 
}
```
(This is the preprint's BibTex. We will provide the official BibTex as soon as the paper is published in the journal.)

# License

The ReceiptVQA dataset is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

# Authors

- Thanh-Phong Le - [21520395@gm.uit.edu.vn](mailto:21520395@gm.uit.edu.vn)

- Trung Le Chi Phan - [21522725@gm.uit.edu.vn](mailto:21522725@gm.uit.edu.vn)

- Nghia Hieu Nguyen - [nghiangh@uit.edu.vn](mailto:nghiangh@uit.edu.vn)

- Kiet Van Nguyen - [kietnv@uit.edu.vn](mailto:kietnv@uit.edu.vn)
# Contact

For any inquiries or feedback regarding the ReceiptVQA dataset and LiGT, please contact [21520395@gm.uit.edu.vn](mailto:21520395@gm.uit.edu.vn) (Thanh-Phong Le)