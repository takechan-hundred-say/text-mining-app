import os
import sys
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

def get_project_root():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def find_mecab_path():
    possible_paths = [
        r'C:\Program Files\MeCab\bin',
        r'C:\Program Files (x86)\MeCab\bin',
    ]
    try:
        import winreg
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\MeCab')
            mecab_path = winreg.QueryValueEx(key, 'path')[0]
            return os.path.join(mecab_path, 'bin')
        except:
            pass
    except:
        pass
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def find_r_path():
    import winreg

    def _parse_version(name):
        try:
            return tuple(int(x) for x in name.lstrip('R-').split('.'))
        except:
            return (0,)

    def _find_rscript(path):
        for candidate in [
            os.path.join(path, 'bin', 'x64', 'Rscript.exe'),
            os.path.join(path, 'bin', 'i386', 'Rscript.exe'),
            os.path.join(path, 'bin', 'Rscript.exe'),
        ]:
            if os.path.exists(candidate):
                return candidate
        return None



    candidates = []

    def _collect_from_registry(hkey, subkey):
        try:
            key = winreg.OpenKey(hkey, subkey)
            try:
                r_path, _ = winreg.QueryValueEx(key, 'InstallPath')
                rscript = _find_rscript(r_path)
                if rscript:
                    candidates.append((_parse_version('0'), rscript))
            except:
                pass
            i = 0
            while True:
                try:
                    version_subkey = winreg.EnumKey(key, i)
                    full_path = f'{subkey}\\{version_subkey}'
                    vkey = winreg.OpenKey(hkey, full_path)
                    try:
                        r_path, _ = winreg.QueryValueEx(vkey, 'InstallPath')
                        rscript = _find_rscript(r_path)
                        if rscript:
                            ver = _parse_version(version_subkey)
                            candidates.append((ver, rscript))
                    except:
                        pass
                    winreg.CloseKey(vkey)
                    i += 1
                except:
                    break
            winreg.CloseKey(key)
        except:
            pass

    _collect_from_registry(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\R-core\R')
    _collect_from_registry(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\R-core\R64')
    _collect_from_registry(winreg.HKEY_CURRENT_USER, r'SOFTWARE\R-core\R')

    try:
        import subprocess
        result = subprocess.run(['where', 'Rscript.exe'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if os.path.exists(line):
                    candidates.append((None, line))
    except:
        pass

    for base in [r'C:\Program Files\R', r'C:\Program Files (x86)\R']:
        if os.path.isdir(base):
            for ver in os.listdir(base):
                full = os.path.join(base, ver)
                if os.path.isdir(full):
                    rscript = _find_rscript(full)
                    if rscript:
                        candidates.append((_parse_version(ver), rscript))

    def _sort_key(item):
        ver, _ = item
        return ver if ver is not None else (0,)

    candidates.sort(key=_sort_key, reverse=True)
    return candidates[0][1] if candidates else None

_FONT_PATH_CANDIDATES = [
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\yugothic.ttc",
    r"C:\Windows\Fonts\YuGothR.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
]

DEFAULT_FONT_PATH = None
for _fp in _FONT_PATH_CANDIDATES:
    if os.path.exists(_fp):
        DEFAULT_FONT_PATH = _fp
        break

if DEFAULT_FONT_PATH:
    from matplotlib import font_manager
    font_manager.fontManager.addfont(DEFAULT_FONT_PATH)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=DEFAULT_FONT_PATH).get_name()

@st.cache_data
def load_polarity_dict_tohoku():
    import unicodedata
    base = get_project_root()
    dict_path = os.path.join(base, 'dic', 'pn.csv.m3.120408.trim')
    polarity_dict = {}
    if not os.path.exists(dict_path):
        st.warning(
            f"辞書ファイルが見つかりません: {dict_path}\n"
            "dicフォルダに `pn.csv.m3.120408.trim` を配置してください。"
        )
        return None
    try:
        df_pncsv = pd.read_csv(
            dict_path,
            sep="\t",
            names=["term", "sentiment", "semantic_category"],
            encoding="utf-8"
        )
        for _, row in df_pncsv.iterrows():
            term_nfkc = unicodedata.normalize('NFKC', str(row['term']))
            polarity_dict[term_nfkc] = str(row['sentiment'])
    except Exception as e:
        st.error(f"東北大学極性辞書の読み込みに失敗しました: {e}")
        return None
    return polarity_dict
