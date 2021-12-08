
class Dataset:
    def __init__(self):
        pass

    def train_img_size(self, csv_file_list):
        _width = 0
        _height = 0

        for csv_file_name in csv_file_list:
            csv_data = pd.read_csv(csv_file_name, sep=';')

            max_width = 0
            max_height = 0

            for img_width in csv_data['Width']:
                if max_width < img_width:
                    max_width = img_width

            for img_height in csv_data['Height']:
                if max_height < img_height:
                    max_height = img_height

            print('{}, max_width={}, max_height={}'.format(csv_file_name, max_width, max_height))

            if _width < max_width:
                _width = max_width
            if _height < max_height:
                _height = max_height

        print('max_width={}, max_height={}'.format(_width, _height))

class TrainDataset(Dataset):

    csv_file_list = [
        './train_images/00000/GT-00000.csv',
        './train_images/00001/GT-00001.csv',
        './train_images/00002/GT-00002.csv',
        './train_images/00003/GT-00003.csv',
        './train_images/00004/GT-00004.csv',
        './train_images/00005/GT-00005.csv',
        './train_images/00006/GT-00006.csv',
        './train_images/00007/GT-00007.csv',
        './train_images/00008/GT-00008.csv',
        './train_images/00009/GT-00009.csv',
        './train_images/00010/GT-00010.csv',
        './train_images/00011/GT-00011.csv',
        './train_images/00012/GT-00012.csv',
        './train_images/00013/GT-00013.csv',
        './train_images/00014/GT-00014.csv',
        './train_images/00015/GT-00015.csv',
        './train_images/00016/GT-00016.csv',
        './train_images/00017/GT-00017.csv',
        './train_images/00018/GT-00018.csv',
        './train_images/00019/GT-00019.csv',
        './train_images/00020/GT-00020.csv',
        './train_images/00021/GT-00021.csv',
        './train_images/00022/GT-00022.csv',
        './train_images/00023/GT-00023.csv',
        './train_images/00024/GT-00024.csv',
        './train_images/00025/GT-00025.csv',
        './train_images/00026/GT-00026.csv',
        './train_images/00027/GT-00027.csv',
        './train_images/00028/GT-00028.csv',
        './train_images/00029/GT-00029.csv',
        './train_images/00030/GT-00030.csv',
        './train_images/00031/GT-00031.csv',
        './train_images/00032/GT-00032.csv',
        './train_images/00033/GT-00033.csv',
        './train_images/00034/GT-00034.csv',
        './train_images/00035/GT-00035.csv',
        './train_images/00036/GT-00036.csv',
        './train_images/00037/GT-00037.csv',
        './train_images/00038/GT-00038.csv',
        './train_images/00039/GT-00039.csv',
        './train_images/00040/GT-00040.csv',
        './train_images/00041/GT-00041.csv',
        './train_images/00042/GT-00042.csv',
    ]

    '''
    ./train_images/00000/GT-00000.csv, max_width=144, max_height=148
    ./train_images/00001/GT-00001.csv, max_width=164, max_height=170
    ./train_images/00002/GT-00002.csv, max_width=171, max_height=171
    ./train_images/00003/GT-00003.csv, max_width=151, max_height=156
    ./train_images/00004/GT-00004.csv, max_width=148, max_height=152
    ./train_images/00005/GT-00005.csv, max_width=161, max_height=168
    ./train_images/00006/GT-00006.csv, max_width=98, max_height=108
    ./train_images/00007/GT-00007.csv, max_width=176, max_height=175
    ./train_images/00008/GT-00008.csv, max_width=147, max_height=148
    ./train_images/00009/GT-00009.csv, max_width=170, max_height=171
    ./train_images/00010/GT-00010.csv, max_width=143, max_height=154
    ./train_images/00011/GT-00011.csv, max_width=216, max_height=203
    ./train_images/00012/GT-00012.csv, max_width=196, max_height=196
    ./train_images/00013/GT-00013.csv, max_width=224, max_height=201
    ./train_images/00014/GT-00014.csv, max_width=224, max_height=219
    ./train_images/00015/GT-00015.csv, max_width=125, max_height=164
    ./train_images/00016/GT-00016.csv, max_width=200, max_height=195
    ./train_images/00017/GT-00017.csv, max_width=120, max_height=127
    ./train_images/00018/GT-00018.csv, max_width=205, max_height=192
    ./train_images/00019/GT-00019.csv, max_width=169, max_height=147
    ./train_images/00020/GT-00020.csv, max_width=166, max_height=152
    ./train_images/00021/GT-00021.csv, max_width=200, max_height=178
    ./train_images/00022/GT-00022.csv, max_width=182, max_height=171
    ./train_images/00023/GT-00023.csv, max_width=226, max_height=213
    ./train_images/00024/GT-00024.csv, max_width=191, max_height=168
    ./train_images/00025/GT-00025.csv, max_width=243, max_height=225
    ./train_images/00026/GT-00026.csv, max_width=208, max_height=191
    ./train_images/00027/GT-00027.csv, max_width=181, max_height=165
    ./train_images/00028/GT-00028.csv, max_width=176, max_height=161
    ./train_images/00029/GT-00029.csv, max_width=170, max_height=182
    ./train_images/00030/GT-00030.csv, max_width=221, max_height=192
    ./train_images/00031/GT-00031.csv, max_width=191, max_height=176
    ./train_images/00032/GT-00032.csv, max_width=135, max_height=133
    ./train_images/00033/GT-00033.csv, max_width=161, max_height=166
    ./train_images/00034/GT-00034.csv, max_width=136, max_height=149
    ./train_images/00035/GT-00035.csv, max_width=155, max_height=151
    ./train_images/00036/GT-00036.csv, max_width=124, max_height=122
    ./train_images/00037/GT-00037.csv, max_width=136, max_height=134
    ./train_images/00038/GT-00038.csv, max_width=140, max_height=147
    ./train_images/00039/GT-00039.csv, max_width=188, max_height=193
    ./train_images/00040/GT-00040.csv, max_width=177, max_height=173
    ./train_images/00041/GT-00041.csv, max_width=131, max_height=144
    ./train_images/00042/GT-00042.csv, max_width=124, max_height=118
    max_width=243, max_height=225
    '''

    def __init__(self):
        pass


