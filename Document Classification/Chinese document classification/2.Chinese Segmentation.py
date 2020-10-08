# Date: 2018 - 05 - 22
# Author: Haoliang Chang
# Data: The data could be found in here：http://thuctc.thunlp.org/
# Reference: https://www.jianshu.com/p/233da896226a

import jieba
import os
import logging
import sys  

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format = '%(asctime)s:%(levelname)s:%(message)s')
    logging.root.setLevel(level = logging.INFO)

    file_path = r'F:\\Data Analysis\\github\\THUCNews\data\\'
    files = ['家居_altogether.txt', '教育_altogether.txt',
             '科技_altogether.txt']

    for file in files:
        # Set the input and output files
        file_to_be_segmented = open(file_path + file, mode = 'r', encoding = 'UTF-8')
        output_file = open(file_path + file + '_segmented.txt', mode = 'w', encoding = 'UTF-8')
        print('The file is opened~')
        lineNum = 1
        line = file_to_be_segmented.readline()
        
        # Segment the words using jieba
        while line:
            print('Line(article) ', lineNum, 'is processing')
            segmentation = jieba.cut(line, cut_all = False)
            line_seg = ' '.join(segmentation)
            output_file.writelines(line_seg)
            lineNum += 1
            line = file_to_be_segmented.readline()

        print('finished', file)
        file_to_be_segmented.close()
        output_file.close()
