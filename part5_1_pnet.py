import os
import part2_nets
import part4_train

if __name__ == '__main__':
    index = 0
    params_path = "Data/Params_2/P"
    max_params_path = "Data/Params_2/P_max"
    doc_path = "Data/Params_2/P.txt"
    root_path = ["G:/MTCNN_Data/Dataset_1/Train_Image",
                 "G:/MTCNN_Data/Dataset_1/Validate_Image"]
    pic_size = 12
    doc_name = [["train_positive.txt", "train_part.txt", "train_negative.txt"],
                ["validate_positive.txt", "validate_part.txt", "validate_negative.txt"],
                ["test_positive.txt", "test_part.txt", "test_negative.txt"]]

    if not os.path.exists(params_path):
        os.makedirs(params_path)
    if not os.path.exists(max_params_path):
        os.makedirs(max_params_path)

    net = part2_nets.PNet()
    save_path = f"{params_path}/p_net"
    max_save_path = f"{max_params_path}/p_net"
    tra_path = f"{root_path[0]}/{str(pic_size)}"
    val_path = f"{root_path[1]}/{str(pic_size)}"
    tra_doc_path = doc_name[0]
    val_doc_path = doc_name[1]
    trainer = part4_train.Trainer(net, save_path, max_save_path,
                                  tra_path, val_path, tra_doc_path, val_doc_path, index, pic_size)
    trainer.train(0.7, 1000, doc_path, False, num=4)
