import numpy as np


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