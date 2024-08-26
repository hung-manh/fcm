import numpy as np
import math


def round_float(number: float) -> float:
    return round(number, 3)


def norm_distances(XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import cdist
    return cdist(XA, XB)
    # return np.sqrt(((XA[:, np.newaxis, :] - XB) ** 2).sum(axis=2))


def extract_labels(U: np.ndarray) -> np.ndarray:
    return np.argmax(U, axis=1)


def extract_clusters(data: np.ndarray, labels: np.ndarray, C: int) -> list:
    return [data[labels == i] for i in range(C)]


def export_to_latex_data(metrics, filename):
    latex_code = r"""
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{FCM} & \textbf{Size} & \textbf{C} & \textbf{Time} & \textbf{DI$\triangledown$} & \textbf{DB$\triangledown$} & \textbf{PC$\Delta$} & \textbf{CE$\triangledown$} & \textbf{S$\triangledown$} & \textbf{CH$\Delta$} & \textbf{SI$\Delta$} & \textbf{FHV$\Delta$} & \textbf{CS$\triangledown$} \\ \hline
"""
    for metric in metrics:
        latex_code += f"{metric['FCM']} & {metric['Size']} & {metric['C']} & {metric['Time']} & {metric['DI']} & {metric['DB']} & {metric['PC']} & {metric['CE']} & {metric['S']} & {metric['CH']} & {metric['SI']} & {metric['FHV']} & {metric['CS']} \\\\ \\hline\n"

    latex_code += r"""
\end{tabular}
"""
    with open(filename, "w") as file:
        file.write(latex_code)
        
def export_to_latex_image(metrics, filename):
    latex_code = r"""
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{FCM} & \textbf{Time} & \textbf{DB$\triangledown$} & \textbf{PC$\Delta$} & \textbf{CE$\triangledown$} & \textbf{S$\triangledown$} & \textbf{CH$\Delta$} & \textbf{FHV$\Delta$} & \textbf{CS$\triangledown$} \\ \hline
"""
    for metric in metrics:
        latex_code += f"{metric['FCM']} & {metric['Time']}  & {metric['DB']} & {metric['PC']} & {metric['CE']} & {metric['S']} & {metric['CH']} & {metric['FHV']} & {metric['CS']} \\\\ \\hline\n"

    latex_code += r"""
\end{tabular}
"""
    with open(filename, "w") as file:
        file.write(latex_code)


def export_to_latex_image_v2(metrics, filename):
    latex_code = r"""
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{SSCFCM} & \textbf{Time} & \textbf{DB$\triangledown$} & \textbf{PC$\Delta$} & \textbf{CE$\triangledown$} & \textbf{S$\triangledown$} & \textbf{CH$\Delta$} & \textbf{SI$\Delta$} & \textbf{FHV$\Delta$} & \textbf{CS$\triangledown$} \textbf{F1$\Delta$} & \textbf{AC$\Delta$} \\ \hline
"""
    for metric in metrics:
        latex_code += f"{metric['SSCFCM']} & {metric['Time']}  & {metric['DB']} & {metric['PC']} & {metric['CE']} & {metric['S']} & {metric['CH']} & {metric['SI']} & {metric['FHV']} & {metric['CS']} & {metric['F1']} & {metric['AC']}\\\\ \\hline\n"

    latex_code += r"""
\end{tabular}
"""
    with open(filename, "w") as file:
        file.write(latex_code)

def export_to_latex_image_v3(metrics, filename):
    latex_code = r"""
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{SSFCM} & \textbf{Time} & \textbf{DB$\triangledown$} & \textbf{PC$\Delta$} & \textbf{CE$\triangledown$} & \textbf{S$\triangledown$} & \textbf{CH$\Delta$} & \textbf{SI$\Delta$} & \textbf{FHV$\Delta$} & \textbf{CS$\triangledown$} \textbf{F1$\Delta$} & \textbf{AC$\Delta$} \\ \hline
"""
    for metric in metrics:
        latex_code += f"{metric['SSFCM']} & {metric['Time']}  & {metric['DB']} & {metric['PC']} & {metric['CE']} & {metric['S']} & {metric['CH']} & {metric['SI']} & {metric['FHV']} & {metric['CS']} & {metric['F1']} & {metric['AC']}\\\\ \\hline\n"

    latex_code += r"""
\end{tabular}
"""
    with open(filename, "w") as file:
        file.write(latex_code)

def random_negative_assignment(labels: np.ndarray, ratio: float = 0.3, val: float = -1) -> np.ndarray:
    _length = len(labels)
    # Tính số lượng phần từ cần gán giá trị -1
    neg_count = int(ratio * _length)
    # Tạo một mảng chỉ số ngẫu nhiên không gặp lại
    random_indices = np.random.choice(_length, neg_count, replace=True)
    # Tạo một bản sao của mảng để không làm thay đổi mảng gốc
    result = np.copy(labels)
    # Gán giá trị -1 vào các vị trí ngẫu nhiên
    result[random_indices] = val
    
    return result

def split_data_for_semi_supervised_learning(data: np.ndarray, labels: np.ndarray, n_sites: int = 3, ratio: float = 0.3, val: float = -1) -> list:
    result = []
    datas = np.array_split(data, n_sites)
    labeled = np.array_split(labels, n_sites)
    for i, data in enumerate(datas):
        labels = random_negative_assignment(labels=labeled[i], ratio=ratio, val=val)
        result.append({'X': data, 'Y': labels})
    return result   

def assign_label(label_string: list) -> list:
    cnt = 0
    result = []
    result.append(0)
    for i in range(1, len(label_string)):
        if label_string[i - 1] != label_string[i]:
            cnt += 1
        result.append(cnt)
    return np.array(result)
        

def create_membership_label(data: np.ndarray, label_string: list, n_sites: int = 3, ratio: float = 0.3, C: int = 3) -> tuple:
    # Convert to number
    labeled_number = assign_label(label_string)
    # split data and labels follow n_sites
    datas = np.array_split(data, n_sites)
    labeled = np.array_split(labeled_number, n_sites)
    u_bar = []

    # select random index to get labeled and create u_bar
    for i in range(n_sites):
        u = np.zeros((len(labeled[i]), C[i]))
        num_random_row = np.random.choice(len(labeled[i]), math.ceil(ratio * len(labeled[i])))
        for j in range(len(labeled[i])):
            if j in num_random_row:
                u[j, labeled[i][j]] = 1
        u_bar.append(u)
    # u_bar = np.array(u_bar)

    return datas, labeled, u_bar

def create_membership_label_v2(data: np.ndarray, label_string: list, n_sites: int = 3, ratio: float = 0.3, C: int = 3) -> tuple:
    # Convert to number
    labeled_number = assign_label(label_string)
    # split data and labels follow n_sites
    datas = np.array_split(data, n_sites)
    labeled = np.array_split(labeled_number, n_sites)
    u_bar = []

    # select random index to get labeled and create u_bar
    for i in range(n_sites):
        u = np.zeros((len(labeled[i]), C))
        num_random_row = np.random.choice(len(labeled[i]), math.ceil(ratio * len(labeled[i])))
        for j in range(len(labeled[i])):
            if j in num_random_row:
                u[j, labeled[i][j]] = 1
        u_bar.append(u)
    u_bar = np.array(u_bar)

    return datas, labeled, u_bar

def create_membership_for_semi_supervised_learning(data: np.ndarray, label_string: list, ratio: float = 0.3, C: int = 3) -> tuple:
    # Convert to number
    labeled_number = assign_label(label_string)
    # select random index to get labeled and create u_bar
    u_bar = np.zeros((len(labeled_number), C))
    num_random_row = np.random.choice(len(labeled_number), math.ceil(ratio * len(labeled_number)))
    for j in range(len(labeled_number)):
        if j in num_random_row:
            u_bar[j, labeled_number[j]] = 1
    u_bar = np.array(u_bar)

    return data, labeled_number, u_bar

def create_membership_for_semi_supervised_learning_image(data: np.ndarray, label: list, ratio: float = 0.3, C: int = 3) -> tuple:
    # # Convert to number
    # # select random index to get labeled and create u_bar
    # u_bar = np.zeros((len(label), C)) # Khởi tạo membership u_bar với shape = (n, C)
    
    # # Khời tạo một mảng có kích thước math.ceil(ratio * len(label)) và giá trị nằm trong khoảng từ 0 đến len(label)
    # num_random_row = np.random.choice(len(label), math.ceil(ratio * len(label)))
    
    # for j in range(len(label)):
    #     if j in num_random_row:
    #         u_bar[j, label[j]] = 1
            
    # u_bar = np.array(u_bar)
    # return data, label, u_bar
    # Hình như mới chỉ gán
    
    u_bar = np.zeros((len(label), C)) # Khởi tạo membership u_bar với shape = (n, C)
    
    num_labeled = math.ceil(ratio * len(label))

    # Khởi tạo một mảng có kích thước math.ceil(ratio * len(label)) và giá trị nằm trong khoảng từ 0 đến len(label), và nếu nó đã có rồi thì không bị ghi đè(replace=False)
    labeled_indices = np.random.choice(len(label), size=num_labeled, replace=False)

    # Gán nhãn cho các điểm ảnh 
    u_bar[labeled_indices, label[labeled_indices]] = 1

    return data, label, u_bar