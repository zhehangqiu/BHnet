### Download the Datasets
- reside-indoor [[Baidu](https://pan.baidu.com/s/1Qk9VX3GLUrX8NRuV0xhv6Q?pwd=0uby)]
test set [ [Baidu](https://pan.baidu.com/s/1R6qWri7sG1hC_Ifj-H6DOQ?pwd=o5sk)]
#### Download the model 
- SOTS-Indoor [[Baidu](https://pan.baidu.com/s/1CE3gCp-3I8KCpB0rBw1SGQ?pwd=cisq)]
- SOTS-Outdoor [[Baidu](https://pan.baidu.com/s/18bsJOd7jZ-NF7CmAO0Q9ig?pwd=3iro)]
#### Testing on SOTS-Indoor
~~~
python main.py --data Indoor --save_image True --mode test --data_dir your_path/reside-indoor --test_model path_to_its_model
~~~
#### Testing on SOTS-Outdoor
~~~
python main.py --data Outdoor --save_image True --mode test --data_dir your_path/reside-outdoor --test_model path_to_ots_model
~~~
