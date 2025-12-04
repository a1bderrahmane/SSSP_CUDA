git clone https://github.com/NVIDIA/cuCollections.git
cd cuCollections && mkdir build && cd build && cmake .. && make -j
cd cuCollections && sudo cp -r include/cuco /usr/local/include/
