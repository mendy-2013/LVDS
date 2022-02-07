import tensorflow as tf
import numpy as np
import time

start = time.clock()

def read_img(path):
    return tf.image.decode_image(tf.read_file(path))

def psnr(tf_img1, tf_img2):
    return tf.image.psnr(tf_img1, tf_img2, max_val=255)

def ssim(tf_img1, tf_img2):
    return tf.image.ssim(tf_img1, tf_img2, max_val=255)

def main(a, b):
    # t1 = read_img('./Data/00001_00_0.1s_gt.png')               #Clean Image
    # t2 = read_img('./Data/00001_00_0.1s_out_optimized.png')       #Noise Image
    t1 = read_img(a)
    t2 = read_img(b)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        p = sess.run(psnr(t1, t2))
        s = sess.run(ssim(t1, t2))
    return p, s

path1 = ''  # 指定输出结果文件夹
path2 = ''  # 指定原图文件夹
list_psnr = []
list_ssim = []
file = open(r'test_a计算结果.txt', mode='w', encoding='utf-8')
for i in range(1,277):
    print('第%s张图片' % i)

    a = path1 +str(i) + '.png'
    b = path2 + str(i)  + '.png'
    p , s = main(a, b)
    list_ssim.append(s)
    list_psnr.append(p)
    file.write('第%s张图片:' % i + '\n')
    file.write('psnr, {:.5f}'.format(p) + '\n')
    file.write('ssim, {:.5f}'.format(s) + '\n')


elapsed = (time.clock() - start)
file.write("\n")
file.write("汇总:\n")
file.write('平均PSNR, {:.5f}'.format(np.mean(list_psnr)) + '\n')
file.write('平均SSIM, {:.5f}'.format(np.mean(list_ssim)) + '\n')
file.write('Time used, {:.5f}'.format(elapsed) + '\n')
file.close()