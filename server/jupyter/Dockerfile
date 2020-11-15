FROM jupyter/datascience-notebook:python-3.8.6

RUN MPL_DATA=/opt/conda/lib/python3.8/site-packages/matplotlib/mpl-data && \
    curl -o ipaexg.zip https://moji.or.jp/wp-content/ipafont/IPAexfont/ipaexg00401.zip && \
    unzip ipaexg.zip && \
		mkdir /opt/conda/fonts && \
    cp ipaexg00401/ipaexg.ttf /opt/conda/fonts/ipaexg.ttf && \
#    echo "font.family: IPAexGothic" >> $MPL_DATA/matplotlibrc && \
		fc-cache -fv && rm -rf ipaexg* .cache/

RUN pip install --quiet --no-cache-dir \
    'eland==7.10.0b1' && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
