cd dataset/
mkdir products
git clone https://github.com/luoxiaojian/xtrapulp.git
cd xtrapulp
make
make libxtrapulp
cd ../
python prepare_products.py 
cd .. 
