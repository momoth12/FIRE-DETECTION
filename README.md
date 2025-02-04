# FIRE-DETECTION

### SatlasPretrain
https://github.com/allenai/satlaspretrain_models/tree/main

### Entrainement

Pour entrainer le modèle, lancer le script ```train_satlas.py```.

```bash
python train_satlas.py \
  --model_id <model identifier str> \
  --fpn <pour inclure ou non le FPN pretrained> \
  --train_backbone --train_fpn <pour finetune ou non les parties pretrained> \
  --load_model <fichier où un modèle est déjà sauvegarder> \
  --model_name <nom du model pour la sauvegarde>

  --num_epochs 50 \
  --batch_size 16 \
  --lr 1e-5 \
  --weight_decay 1e-5 \
```

### Ablation study

Composantes:
*  ```model_id```: différents backbones pré-entrainés du modèle. Voir le github au-dessus, en se limitant au modèles single image RGB [...]_SI_RGB
*  ```fpn```: est-ce qu'on utilise le Feature Pyramid Network pré-entrainé ?
*  ```train_backbone```/```train_fpn```: est-ce qu'on finetune les modèles, ou est-ce qu'on se contente d'entrainer une tête de classification ?

Pour que les résultats soient cohérents, le split train-val des données est fixé.
