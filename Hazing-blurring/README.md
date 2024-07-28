### Download the Datasets
- Gopro-haze [[Baidu](https://pan.baidu.com/s/1i95Nqctf_XMijfRaqKm8eA?pwd=oix0)]
#### Download the model 
- BLURHAZE[[Baidu](https://pan.baidu.com/s/1Qm2CYlqXwxADeIBU8ab9CA?pwd=82ll)]
#### Testing on blurhaze
~~~
python main.py --data GOPRO --mode test --data_dir your_path/blurhaze --test_model path_to_gopro_model --save_image True
~~~