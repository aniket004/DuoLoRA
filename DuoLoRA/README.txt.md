# DuoLoRA: Cycle-Consistent and Rank-Disentangled Content-Style Personalization

DuoLoRA introduces personalized text-to-image generation by disentangling **content** and **style** using **cycle consistency** and **rank-decomposed LoRA modules**. This unofficial repository (for academic purpose) includes scripts for training, inference, and evaluation.

📄 **Paper (ICCV 2025)**:  
https://arxiv.org/pdf/2504.13206

---

### 1. Installation

```bash
git clone https://github.com/aniket004/DuoLoRA.git
cd DuoLoRA
pip install -r requirements.txt
```

---

### 2. Evaluation on Single Object and Style

- Single content and single style image evaluation with DuoLoRA using LayerPrior  
- Generate content-LoRA and style-LoRA independently

```bash
bash train_ziplora_c.sh
bash train_ziplora_s.sh
bash train_rank_layer_prior_cycle_rev_c_s.sh
```

---

### 3. Inference

- Set the DuoLoRA checkpoint path: `$MERGED_LORA_PATH`  
- Set the output directory: `$OUT_DIR`  
- Generate images using a custom prompt

```python
python inference_composite.py --ziplora_name_or_path $MERGED_LORA_PATH --prompt "a sbu dog in szn style running" --output_dir $OUT_DIR
```

---

### 4. Evaluation

```bash
bash evaluate.sh $OUT_DIR
```

---

### 🤝 Acknowledgements

This repository builds upon and was inspired by prior open-source efforts, including:  
- **ZipLoRA (PyTorch)**: https://github.com/mkshing/ziplora-pytorch

---

### 📫 Contact

For questions, issues, or collaboration, please reach out to:  
**Aniket Roy** — ank.roy4@gmail.com

---

### 📌 Citation

If you use DuoLoRA in your research, please cite:

```bibtex
@article{roy2025duolora,
  title={DuoLoRA: Cycle-consistent and Rank-disentangled Content-Style Personalization},
  author={Roy, Aniket and Borse, Shubhankar and Kadambi, Shreya and Das, Debasmit and Mahajan, Shweta and Garrepalli, Risheek and Park, Hyojin and Nayak, Ankita and Chellappa, Rama and Hayat, Munawar and others},
  journal={ICCV},
  year={2025}
}
```

---
