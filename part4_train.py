import os
import torch
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from torch.utils.data import DataLoader
from part3_get_sample import FaceDataset
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, net, save_path, max_save_path, tra_dataset_path, val_dataset_path,
                 tra_doc_path, val_doc_path, index, face_size):
        self.save_path = save_path
        self.max_save_path = max_save_path
        self.tra_dataset_path = tra_dataset_path
        self.tra_doc_path = tra_doc_path
        self.val_dataset_path = val_dataset_path
        self.val_doc_path = val_doc_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = net.to(self.device)
        self.index = index
        self.face_size = face_size

        self.con_loss_fn = torch.nn.BCELoss()
        self.loc_loss_fn = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        # self.optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.5, weight_decay=0.0005)

        self.old_path = f"{save_path}_{self.index}.pth"
        if os.path.exists(self.old_path):
            self.net.load_state_dict(torch.load(self.old_path))
            print("Load Params Successful...")
        else:
            print("No Params...")

    def train(self, alpha, bat_size, doc_path, is_mark, num=14):
        doc = open(doc_path, "a+")
        face_dataset = FaceDataset(self.tra_dataset_path, self.tra_doc_path, 0, is_mark)
        dataloader = DataLoader(face_dataset, batch_size=bat_size, shuffle=True)

        val_dataset = FaceDataset(self.val_dataset_path, self.val_doc_path, 1, is_mark)
        # print(len(face_dataset), len(val_dataset))
        # exit()
        val_dataloader = DataLoader(val_dataset, batch_size=bat_size // 2, shuffle=True)
        # summary_writer = SummaryWriter(f"Data/log_/logs_{self.face_size}")

        epoch = 0
        start_num = 0
        avg_num = 5
        max_r2 = 0.70  # P
        # max_r2 = 0.886895443182027  # R
        # max_r2 = 0.70  # O

        tra_loss_total = []
        val_loss_total = []

        tra_r2_total = []
        val_r2_total = []

        tra_acc_total = []
        val_acc_total = []

        while True:
            print()
            tra_loss_list = []
            tra_r2_list = []
            tra_acc_list = []
            self.net.train()
            for i, (img_data, con_, loc_) in enumerate(dataloader):
                print(i + 1)
                img_data = img_data.to(self.device)
                con_ = con_.to(self.device)
                loc_ = loc_.to(self.device)

                output_con_, output_loc_ = self.net(img_data)
                output_con_, output_loc_ = output_con_.view(-1, 1), output_loc_.view(-1, num)

                """计算分类的损失"""
                "eq:等于，lt:小于，gt:大于，le:小于等于，ge:大于等于"
                con_mask = torch.lt(con_, 2)  # 得到分类标签小于2的bool值，a<2, [0, 1, 2]-->[True, True, False]
                con = torch.masked_select(con_, con_mask)  # 通过掩码，得到符合条件的置信度
                output_con = torch.masked_select(output_con_, con_mask)  # 得到符合条件值的置信度输出值
                # print(output_con.shape, con.shape)
                # exit()
                con_loss = self.con_loss_fn(output_con, con)  # 置信度损失

                """计算坐标损失"""
                loc_mask = torch.gt(con_, 0)  # 得到分类标签大于0的bool值，a>0，[0, 1, 2]-->[False, True, True]
                loc = torch.masked_select(loc_, loc_mask)  # 得到符合条件的偏移量标签值
                output_loc = torch.masked_select(output_loc_, loc_mask)  # 得到符合条件的偏移量输出值
                loc_loss = self.loc_loss_fn(output_loc, loc)  # 偏移量损失

                loss = alpha * con_loss + (1 - alpha) * loc_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                tra_loss_list.append(loss.cpu().item())

                tra_out_location = output_loc.detach().cpu().numpy()
                tra_label_location = loc.detach().cpu().numpy()
                tra_r2 = r2_score(tra_out_location, tra_label_location)
                tra_r2_list.append(tra_r2)

                output_con[output_con >= 0.5] = 1.0
                output_con[output_con < 0.5] = 0.0
                # tra_acc = (torch.sum(torch.eq(output_con, con)) / bat_size).item()
                tra_acc = accuracy_score(output_con.detach().cpu().numpy(), con.detach().cpu().numpy())
                tra_acc_list.append(tra_acc)

                if (i + 1) % 50 == 0:
                    print(f"{epoch + self.index + 1} | "
                          f"train_loss:{np.mean(tra_loss_list)} | "
                          f"train_acc:{np.mean(tra_acc_list)} | "
                          f"train_r2:{np.mean(tra_r2_list)}")

                if (i + 1) % 50 == 0:
                    tra_loss_total.append(np.mean(tra_loss_list))
                    tra_r2_total.append(np.mean(tra_r2_list))
                    tra_acc_total.append(np.mean(tra_acc_list))

                    self.net.eval()
                    val_loss_list = []
                    val_r2_list = []
                    val_acc_list = []
                    for v_img_data, v_con_, v_loc_ in val_dataloader:
                        v_img_data = v_img_data.to(self.device)
                        v_con_ = v_con_.to(self.device)
                        v_loc_ = v_loc_.to(self.device)

                        v_output_con_, v_output_loc_ = self.net(v_img_data)
                        v_output_con_, v_output_loc_ = v_output_con_.view(-1, 1), v_output_loc_.view(-1, num)

                        """计算分类的损失"""
                        "eq:等于，lt:小于，gt:大于，le:小于等于，ge:大于等于"
                        v_con_mask = torch.lt(v_con_, 2)  # 得到分类标签小于2的bool值，a<2, [0, 1, 2]-->[True, True, False]
                        v_con = torch.masked_select(v_con_, v_con_mask)  # 通过掩码，得到符合条件的置信度
                        v_output_con = torch.masked_select(v_output_con_, v_con_mask)  # 得到符合条件值的置信度输出值
                        v_con_loss = self.con_loss_fn(v_output_con, v_con)  # 置信度损失

                        """计算坐标损失"""
                        v_loc_mask = torch.gt(v_con_, 0)  # 得到分类标签大于0的bool值，a>0，[0, 1, 2]-->[False, True, True]
                        v_loc = torch.masked_select(v_loc_, v_loc_mask)  # 得到符合条件的偏移量标签值
                        v_output_loc = torch.masked_select(v_output_loc_, v_loc_mask)  # 得到符合条件的偏移量输出值
                        v_loc_loss = self.loc_loss_fn(v_output_loc, v_loc)  # 偏移量损失

                        v_loss = alpha * v_con_loss + (1 - alpha) * v_loc_loss
                        val_loss_list.append(v_loss.cpu().item())
                        val_out_location = v_output_loc.detach().cpu().numpy()
                        val_label_location = v_loc.detach().cpu().numpy()
                        val_r2 = r2_score(val_out_location, val_label_location)
                        val_r2_list.append(val_r2)

                        v_output_con[v_output_con >= 0.5] = 1.0
                        v_output_con[v_output_con < 0.5] = 0.0
                        # val_acc = (torch.sum(torch.eq(v_output_con, v_con)) / bat_size).item()
                        val_acc = accuracy_score(v_output_con.detach().cpu().numpy(), v_con.detach().cpu().numpy())
                        val_acc_list.append(val_acc)

                    val_loss_total.append(np.mean(val_loss_list))
                    val_r2_total.append(np.mean(val_r2_list))
                    val_acc_total.append(np.mean(val_acc_list))
                    print(f"{epoch + self.index + 1} | "
                          f"val_loss:{np.mean(val_loss_list)} | "
                          f"val_acc:{np.mean(val_acc_list)} | "
                          f"val_r2:{np.mean(val_r2_list)}")

                    '''
                    train_avg_acc = tra_acc_total[epoch]
                    train_avg_loss = tra_loss_total[epoch]
                    train_avg_r2 = tra_r2_total[epoch]

                    val_avg_acc = val_acc_total[epoch]
                    val_avg_loss = val_loss_total[epoch]
                    val_avg_r2 = val_r2_total[epoch]

                    summary_writer.add_scalars("loss", {"train_loss": train_avg_loss, "val_loss": val_avg_loss},
                                               self.index + epoch + 1)
                    summary_writer.add_scalars("acc", {"train_acc": train_avg_acc, "val_acc": val_avg_acc},
                                               self.index + epoch + 1)
                    summary_writer.add_scalars("r2", {"train_r2": train_avg_r2, "val_r2": val_avg_r2},
                                               self.index + epoch + 1)
                    '''

                    doc.write(f"{epoch + self.index + 1} | "
                              f"val_loss:{np.mean(val_loss_list)} | "
                              f"val_acc:{np.mean(val_acc_list)} | "
                              f"val_r2:{np.mean(val_r2_list)}\n")

                    torch.save(self.net.state_dict(), f"{self.save_path}_{self.index + epoch + 1}.pth")

                    if np.mean(val_r2_list) > max_r2:
                        max_r2 = np.mean(val_r2_list)
                        torch.save(self.net.state_dict(), f"{self.max_save_path}_{self.index + epoch + 1}.pth")
                        print(f"[{self.face_size}]: The params is saved successfully...")

                    # 过拟合判断
                    r2_length = len(val_r2_total)
                    if (r2_length - start_num) % 10 == 0:
                        start_r2 = val_r2_total[start_num:r2_length - avg_num]
                        end_r2 = val_r2_total[r2_length - avg_num:r2_length]
                        if np.mean(start_r2) > np.mean(end_r2):
                            doc.close()
                            exit()
                        start_num += avg_num
                    # summary_writer.close()
                    epoch += 1
                    tra_loss_list = []
                    tra_r2_list = []
                    tra_acc_list = []

            # self.net.eval()
            # val_loss_list = []
            # val_r2_list = []
            # val_acc_list = []
            # for v_img_data, v_con_, v_loc_ in val_dataloader:
            #     v_img_data = v_img_data.to(self.device)
            #     v_con_ = v_con_.to(self.device)
            #     v_loc_ = v_loc_.to(self.device)
            #
            #     v_output_con_, v_output_loc_ = self.net(v_img_data)
            #     v_output_con_, v_output_loc_ = v_output_con_.view(-1, 1), v_output_loc_.view(-1, 4)
            #
            #     """计算分类的损失"""
            #     "eq:等于，lt:小于，gt:大于，le:小于等于，ge:大于等于"
            #     v_con_mask = torch.lt(v_con_, 2)  # 得到分类标签小于2的bool值，a<2, [0, 1, 2]-->[True, True, False]
            #     v_con = torch.masked_select(v_con_, v_con_mask)  # 通过掩码，得到符合条件的置信度
            #     v_output_con = torch.masked_select(v_output_con_, v_con_mask)  # 得到符合条件值的置信度输出值
            #     # print(output_con.shape, con.shape)
            #     # exit()
            #     v_con_loss = self.con_loss_fn(v_output_con, v_con)  # 置信度损失
            #
            #     """计算坐标损失"""
            #     v_loc_mask = torch.gt(v_con_, 0)  # 得到分类标签大于0的bool值，a>0，[0, 1, 2]-->[False, True, True]
            #     v_loc = torch.masked_select(v_loc_, v_loc_mask)  # 得到符合条件的偏移量标签值
            #     v_output_loc = torch.masked_select(v_output_loc_, v_loc_mask)  # 得到符合条件的偏移量输出值
            #     v_loc_loss = self.loc_loss_fn(v_output_loc, v_loc)  # 偏移量损失
            #
            #     v_loss = alpha * v_con_loss + (1 - alpha) * v_loc_loss
            #
            #     self.optimizer.zero_grad()
            #     v_loss.backward()
            #     self.optimizer.step()
            #
            #     val_loss_list.append(v_loss.cpu().item())
            #
            #     val_out_location = v_output_loc.detach().cpu().numpy()
            #     val_label_location = v_loc.detach().cpu().numpy()
            #     val_r2 = r2_score(val_out_location, val_label_location)
            #     val_r2_list.append(val_r2)
            #
            #     v_output_con[v_output_con >= 0.5] = 1.0
            #     v_output_con[v_output_con < 0.5] = 0.0
            #     # val_acc = (torch.sum(torch.eq(v_output_con, v_con)) / bat_size).item()
            #     val_acc = accuracy_score(v_output_con.detach().cpu().numpy(), v_con.detach().cpu().numpy())
            #     val_acc_list.append(val_acc)
            #
            # val_loss_total.append(np.mean(val_loss_list))
            # val_r2_total.append(np.mean(val_r2_list))
            # val_acc_total.append(np.mean(val_acc_list))
            #
            # print(f"Val_loss：{np.mean(val_loss_list)}  |  "
            #       f"Val_R2_score：{np.mean(val_r2_list)} | "
            #       f"Val_acc：{np.mean(val_acc_list)}")
            #
            # # 模型保存
            # if np.mean(val_r2_list) > max_r2:
            #     max_r2 = np.mean(val_r2_list)
            #     torch.save(self.net.state_dict(), f"{self.save_path}/{index + epoch + 1}.pth")
            #     print("The params is saved successfully...")
            #
            # # 过拟合判断
            # r2_length = len(val_r2_total)
            # if (r2_length - start_num) % 10 == 0:
            #     start_r2 = val_r2_total[start_num:r2_length - avg_num]
            #     end_r2 = val_r2_total[r2_length - avg_num:r2_length]
            #     if np.mean(start_r2) > np.mean(end_r2):
            #         print(tra_loss_total, tra_acc_total, tra_r2_total)
            #         print(val_loss_total, val_acc_total, val_r2_total)
            #         exit()
            #     start_num += avg_num
