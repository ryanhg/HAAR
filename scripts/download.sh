# download neural haircut strand prior model
if [ ! -d "strand_prior" ]; then
  gdown --folder https://drive.google.com/drive/folders/1TCdJ0CKR3Q6LviovndOkJaKm8S1T9F_8
fi

# download HAAR ckpt:
if [ ! -d "ckpt" ]; then
  gdown --folder https://drive.google.com/drive/folders/1duKjnQT_j-ZBR5pzgN4s1_a7iRMm-1Sp
fi

# download HAAR scalp and textures data:
if [ ! -d "data" ]; then
  gdown --folder https://drive.google.com/drive/folders/1UB8XfVlr4oxT71Y67WykozCGgNWqQYD9
fi

# download HAAR examples (optional):
if [ ! -d "examples" ]; then
  gdown --folder https://drive.google.com/drive/folders/1IqlrFof9tAis6q8aWySrbfKjtTVN6mJx
fi

if [ ! -d "models" ]; then
  gdown --folder https://drive.google.com/drive/folders/1OnJsOGGj8A8bcAWGYA6r-KW1YfpVYvJ1
fi