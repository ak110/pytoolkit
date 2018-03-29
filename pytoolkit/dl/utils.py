"""DeepLearning(主にKeras)関連。"""
import re


def parse_keras_output(output_text: str):
    """Kerasのコンソール出力からlossなどを見つけてDataFrameに入れて返す。結果は(列名, DataFrame)の配列。"""
    import pandas as pd
    try:
        pat = re.compile(r'- (\w+): ([-+\.e\d]+)')
        data = {}
        columns = []  # キー(出てきた順)
        for line in output_text.split('\n'):
            if '- ETA:' in line:
                continue

            for m in pat.finditer(line):
                key = m.group(1)
                value = float(m.group(2))
                if key not in data:
                    data[key] = []
                    columns.append(key)
                data[key].append(value)

        if len(columns) == 0:
            return []

        df = pd.DataFrame.from_dict(data)
        df.index += 1

        df_list = []
        for col in df.columns:
            if col.startswith('val_') and col[4:] in df.columns:
                continue
            # acc, val_accなどはまとめる。
            targets = [col]
            val_col = 'val_' + col
            if val_col in df.columns:
                targets.append(val_col)
            df_list.append((col, df[targets]))

        return df_list
    except ValueError:
        return []
