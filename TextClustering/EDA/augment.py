# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

from EDA import *

# arguments to be parsed from command line
import argparse

from EDA.eda import eda

# generate more data with standard augmentation
def gen_eda(train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
    writer = open(output_file, 'w')
    lines = open(train_orig, 'r').readlines()

    for i, line in enumerate(lines):
        # parts = line[:-1].split('\t')
        # label = parts[0]
        sentence = line
        aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd,
                            num_aug=num_aug)
        for aug_sentence in aug_sentences:
            writer.write(aug_sentence + '\n')

    writer.close()
    print("generated augmented sentences with EDA for " + train_orig + " to " + output_file + " with num_aug=" + str(
        num_aug))
