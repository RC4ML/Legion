cd dataset/
mkdir paper100m
git clone https://github.com/luoxiaojian/xtrapulp.git
cd xtrapulp
make
make libxtrapulp
cd ../
python prepare_paper100m.py 
cd .. 
