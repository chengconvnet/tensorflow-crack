from pathlib import Path
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.keras import models


def mkdir(out_dir):
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()


class Size:
    @staticmethod
    def ceil(x, patch):
        n_patch = np.ceil(x/patch).astype('int32')
        return patch * n_patch

    @staticmethod
    def resize(w, h, patch_w, patch_h):
        w = Size.ceil(w, patch_w)
        h = Size.ceil(h, patch_h)
        return w, h


class PaddingImage:
    def __init__(self, root, match_mode):
        self.root = Path(root)  # 数据根目录
        # 依据给定的模板获取图片
        self.filenames = [filename for filename in self.root.glob(match_mode)]

    def padding(self, filename, sliding_size):
        '''对图片 padding 0'''
        im = Image.open(filename)
        W, H = Size.resize(*im.size, *sliding_size)
        origin_im = Image.new('RGB', (W, H))
        origin_im.paste(im)
        return origin_im

    def save(self, save_dir, sliding_size):
        '''保存 padding 后的图片'''
        mkdir(save_dir)
        _save_dir = Path(save_dir)
        for filename in self.filenames:
            print(filename)
            im = self.padding(filename, sliding_size)
            im.save(_save_dir/filename.name)
            im.close()


class SlidingWindow:
    def __init__(self, filename, sliding_size):
        self.im = Image.open(filename)
        self.sliding_size = sliding_size

    def slice_window(self, row, column):
        w, h = self.sliding_size
        bbox = np.array([0, 0, w, h]) + np.array([w*row, h*column]*2)
        return bbox

    def crop(self, row, column):
        bbox = self.slice_window(row, column)
        return self.im.crop(bbox)

    @property
    def width(self):
        return self.im.size[0]//self.sliding_size[0]

    @property
    def height(self):
        return self.im.size[1]//self.sliding_size[1]

    def meshgrid(self):
        X, Y = np.mgrid[0:self.width, 0:self.height]
        return X, Y

    def scores(self, predict_ys):
        '''转换到原图的 patch 得分
        :param width: window 的宽度
        :param height: window 的高度
        '''
        return predict_ys.reshape(self.width, self.height)

    def __iter__(self):
        for n_w in np.arange(self.width):
            for n_h in np.arange(self.height):
                x = self.crop(n_w, n_h)
                yield np.expand_dims(x, 0)

    def save(self, out_dir):
        mkdir(out_dir)
        out_dir = Path(out_dir)
        filename = Path(self.im.filename).name
        self.im.save(out_dir/filename)

    def mask(self, patch, beta):
        '''beta 不透明度
        '''
        im = Image.new('RGB', patch.size, 'red')
        return Image.blend(patch, im, beta)

    def paste(self, mask, index):
        '''剪切并粘贴 patch'''
        bbox = self.slice_window(*index)
        self.im.paste(mask, bbox)


class SlidingModel:
    def __init__(self, tune_model_dir, model_name):
        '''二分类滑窗模型
        '''
        self._tune_model_dir = tune_model_dir
        self.model_name = model_name

    @property
    def model(self):
        name = f'{self._tune_model_dir}/{self.model_name}.h5'
        return models.load_model(name)

    def window(self, filename, sliding_size):
        return SlidingWindow(filename, sliding_size)

    def __call__(self, window, batch_size=32):
        '''获取预测标签'''
        xs = tf.data.Dataset.from_tensor_slices([x for x in window])
        ys = self.model.predict(xs, batch_size=batch_size)
        return ys

    def positive_indexes(self, scores, alpha):
        '''获取正样本索引'''
        # scores = self.scores(ys)
        return np.argwhere(scores > alpha)

    def mkdir_save_dir(self, save_dir, mask_dir):
        mkdir(save_dir)
        save_dir = f"{save_dir}/{self.model_name}"
        mkdir(save_dir)
        return save_dir

    def mask_name(self, mask_dir, index):
        w, h = index
        name = f"{mask_dir}_{h}_{w}.jpg"
        return name

    def mask_path(self, index, save_dir, mask_dir):
        save_dir = f"{save_dir}/{mask_dir}"
        mkdir(save_dir)
        _mask_name = self.mask_name(mask_dir, index)
        save_path = f"{save_dir}/{_mask_name}"
        return save_path


class CrackModel(SlidingModel):
    def __init__(self, tune_model_dir, model_name):
        super().__init__(tune_model_dir, model_name)

    def predict(self, window, row, column):
        '''预测单张图片
        '''
        x = window.crop(row, column)
        x = np.expand_dims(x, 0)
        # x = tf.constant(x)
        y = self.model.predict(x)
        return y

    def get_class_name(self, ys, alpha=0.8):
        '''alpha 表示最大概率'''
        n_positive = sum(ys > alpha)
        class_name = 'crack' if n_positive >= 2 else 'noncrack'
        return class_name

    def mask_dir(self, filename):
        name = Path(filename).name
        return name.replace('.JPG', '')

    def save_result(self, class_file_names, class_name, save_dir):
        text = '\n'.join(class_file_names)
        _out_dir = f"{save_dir}/{class_name}"
        with open(_out_dir, 'w') as fp:
            fp.write(text)

    def write(self, out_dir, crack_names, non_crack_names):
        '''保存预测后的文件名'''
        self.save_result(crack_names, 'crack.csv', out_dir)
        self.save_result(non_crack_names, 'noncrack.csv', out_dir)

    def run(self, alpha, beta, batch_size, sliding_size, data_dir, save_root):
        crack_names = []
        non_crack_names = []
        for filename in Path(data_dir).iterdir():
            mask_dir = self.mask_dir(filename)
            save_dir = self.mkdir_save_dir(save_root, mask_dir)
            window = self.window(filename, sliding_size)
            ys = self(window, batch_size)
            class_name = self.get_class_name(ys, alpha)
            mask_name = filename.as_posix()
            if class_name == 'crack':
                crack_names.append(mask_name)
                scores = window.scores(ys)
                crack_indexes = self.positive_indexes(scores, alpha)
                for index in crack_indexes:
                    mask_path = self.mask_path(index, save_dir, mask_dir)
                    patch = window.crop(*index)
                    patch.save(mask_path)  # 保存 patch
                    mask = window.mask(patch, beta)
                    window.paste(mask, index)
            else:
                non_crack_names.append(mask_name)
            window.save(save_dir)
        self.write(save_dir, crack_names, non_crack_names)

