# 타이타닉 머신러닝 과제

오늘은 캐글에서 입문자용으로 불리는 타이타닉 생존자 데이터를 가지고 모델훈련을 해보려고 한다

## 문제
타이타닉 데이터 세트를 해결하여라. 목표는 다른 열을 바탕으로 Survived열을 예상하는 것이다.

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "5차 과제_2019250056 최유안",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "nLMVpfvWjIm_"
      },
      "outputs": [],
      "source": [
        "import sys\n",
        "assert sys.version_info >= (3, 7)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "DT97rwviq_E-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import sklearn\n",
        "assert sklearn.__version__ >= \"1.0.1\""
      ],
      "metadata": {
        "id": "wWqubEZbjMda"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "plt.rc('font', size=14)\n",
        "plt.rc('axes', labelsize=14, titlesize=14)\n",
        "plt.rc('legend', fontsize=14)\n",
        "plt.rc('xtick', labelsize=10)\n",
        "plt.rc('ytick', labelsize=10)"
      ],
      "metadata": {
        "id": "Bk-15zBGjOTt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from pathlib import Path\n",
        "import pandas as pd\n",
        "import tarfile\n",
        "import urllib.request\n",
        "\n",
        "def load_titanic_data():\n",
        "    tarball_path = Path(\"datasets/titanic.tgz\")\n",
        "    if not tarball_path.is_file():\n",
        "        Path(\"datasets\").mkdir(parents=True, exist_ok=True)\n",
        "        url = \"https://github.com/ageron/data/raw/main/titanic.tgz\"\n",
        "        urllib.request.urlretrieve(url, tarball_path)\n",
        "        with tarfile.open(tarball_path) as titanic_tarball:\n",
        "            titanic_tarball.extractall(path=\"datasets\")\n",
        "    return [pd.read_csv(Path(\"datasets/titanic\") / filename)\n",
        "            for filename in (\"train.csv\", \"test.csv\")]"
      ],
      "metadata": {
        "id": "tLhkbZW9jPXs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data, test_data = load_titanic_data()"
      ],
      "metadata": {
        "id": "ZNgxDN1mjX-I"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "ilsumaVijce1",
        "outputId": "ee489d62-638d-4679-fc82-e2009b152928"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   PassengerId  Survived  Pclass  \\\n",
              "0            1         0       3   \n",
              "1            2         1       1   \n",
              "2            3         1       3   \n",
              "3            4         1       1   \n",
              "4            5         0       3   \n",
              "\n",
              "                                                Name     Sex   Age  SibSp  \\\n",
              "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
              "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
              "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
              "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
              "4                           Allen, Mr. William Henry    male  35.0      0   \n",
              "\n",
              "   Parch            Ticket     Fare Cabin Embarked  \n",
              "0      0         A/5 21171   7.2500   NaN        S  \n",
              "1      0          PC 17599  71.2833   C85        C  \n",
              "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
              "3      0            113803  53.1000  C123        S  \n",
              "4      0            373450   8.0500   NaN        S  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-f5b20fd6-6853-4b2b-af2f-c84baa4839a3\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>PassengerId</th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Name</th>\n",
              "      <th>Sex</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Ticket</th>\n",
              "      <th>Fare</th>\n",
              "      <th>Cabin</th>\n",
              "      <th>Embarked</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Braund, Mr. Owen Harris</td>\n",
              "      <td>male</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>A/5 21171</td>\n",
              "      <td>7.2500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
              "      <td>female</td>\n",
              "      <td>38.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>PC 17599</td>\n",
              "      <td>71.2833</td>\n",
              "      <td>C85</td>\n",
              "      <td>C</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>Heikkinen, Miss. Laina</td>\n",
              "      <td>female</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>STON/O2. 3101282</td>\n",
              "      <td>7.9250</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
              "      <td>female</td>\n",
              "      <td>35.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>113803</td>\n",
              "      <td>53.1000</td>\n",
              "      <td>C123</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>Allen, Mr. William Henry</td>\n",
              "      <td>male</td>\n",
              "      <td>35.0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>373450</td>\n",
              "      <td>8.0500</td>\n",
              "      <td>NaN</td>\n",
              "      <td>S</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-f5b20fd6-6853-4b2b-af2f-c84baa4839a3')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-f5b20fd6-6853-4b2b-af2f-c84baa4839a3 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-f5b20fd6-6853-4b2b-af2f-c84baa4839a3');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_data = train_data.set_index(\"PassengerId\")\n",
        "test_data = test_data.set_index(\"PassengerId\")"
      ],
      "metadata": {
        "id": "yKadgu7jjmQV"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_data.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bn8PCVw4jpas",
        "outputId": "4d0d11ea-08e4-47fa-ec2a-15960b84e3d2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "Int64Index: 891 entries, 1 to 891\n",
            "Data columns (total 11 columns):\n",
            " #   Column    Non-Null Count  Dtype  \n",
            "---  ------    --------------  -----  \n",
            " 0   Survived  891 non-null    int64  \n",
            " 1   Pclass    891 non-null    int64  \n",
            " 2   Name      891 non-null    object \n",
            " 3   Sex       891 non-null    object \n",
            " 4   Age       714 non-null    float64\n",
            " 5   SibSp     891 non-null    int64  \n",
            " 6   Parch     891 non-null    int64  \n",
            " 7   Ticket    891 non-null    object \n",
            " 8   Fare      891 non-null    float64\n",
            " 9   Cabin     204 non-null    object \n",
            " 10  Embarked  889 non-null    object \n",
            "dtypes: float64(2), int64(4), object(5)\n",
            "memory usage: 83.5+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_data[train_data[\"Sex\"]==\"female\"][\"Age\"].median()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QZRbACe8jqls",
        "outputId": "5424e0c1-7f93-4591-acd7-5ffe3b0707bf"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "27.0"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_data.describe()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "id": "RKpC3_rAjsGW",
        "outputId": "f7f6770b-0b52-4e60-85e4-bc6f4c08b925"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "         Survived      Pclass         Age       SibSp       Parch        Fare\n",
              "count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000\n",
              "mean     0.383838    2.308642   29.699113    0.523008    0.381594   32.204208\n",
              "std      0.486592    0.836071   14.526507    1.102743    0.806057   49.693429\n",
              "min      0.000000    1.000000    0.416700    0.000000    0.000000    0.000000\n",
              "25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400\n",
              "50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200\n",
              "75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000\n",
              "max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-a98acdae-6366-4864-a7f0-c9d11787fd7b\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "      <th>Pclass</th>\n",
              "      <th>Age</th>\n",
              "      <th>SibSp</th>\n",
              "      <th>Parch</th>\n",
              "      <th>Fare</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>714.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "      <td>891.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>0.383838</td>\n",
              "      <td>2.308642</td>\n",
              "      <td>29.699113</td>\n",
              "      <td>0.523008</td>\n",
              "      <td>0.381594</td>\n",
              "      <td>32.204208</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>0.486592</td>\n",
              "      <td>0.836071</td>\n",
              "      <td>14.526507</td>\n",
              "      <td>1.102743</td>\n",
              "      <td>0.806057</td>\n",
              "      <td>49.693429</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.416700</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>20.125000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>7.910400</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>28.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>14.454200</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>38.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>31.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>80.000000</td>\n",
              "      <td>8.000000</td>\n",
              "      <td>6.000000</td>\n",
              "      <td>512.329200</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-a98acdae-6366-4864-a7f0-c9d11787fd7b')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-a98acdae-6366-4864-a7f0-c9d11787fd7b button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-a98acdae-6366-4864-a7f0-c9d11787fd7b');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_data[\"Survived\"].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_Th5B8iEjtSu",
        "outputId": "a8b1b6d2-aff7-4831-c659-f00522f2863e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0    549\n",
              "1    342\n",
              "Name: Survived, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_data[\"Pclass\"].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "x1TMfiW-ju3g",
        "outputId": "eefa6830-734e-40e2-e7a0-9a7d046ca4c3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "3    491\n",
              "1    216\n",
              "2    184\n",
              "Name: Pclass, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_data[\"Sex\"].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "V3BYrA28jwNd",
        "outputId": "217381c9-c5ef-46c4-ccd1-d504e8f6eb75"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "male      577\n",
              "female    314\n",
              "Name: Sex, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_data[\"Embarked\"].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6-lLN6G3jxPR",
        "outputId": "b56422ce-fb80-4576-b2e1-dd8ca7cc99e3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "S    644\n",
              "C    168\n",
              "Q     77\n",
              "Name: Embarked, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "num_pipeline = Pipeline([\n",
        "                         (\"imputer\", SimpleImputer(strategy=\"median\")),\n",
        "                         (\"scaler\", StandardScaler())\n",
        "])"
      ],
      "metadata": {
        "id": "3hG8Y9KrjykD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder"
      ],
      "metadata": {
        "id": "yg98tmnaj0D7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cat_pipeline = Pipeline([\n",
        "                         (\"ordinal_encoder\", OrdinalEncoder()),\n",
        "                         (\"imputer\", SimpleImputer(strategy=\"most_frequent\")),\n",
        "                         (\"cat_encoder\", OneHotEncoder(sparse=False)),\n",
        "])"
      ],
      "metadata": {
        "id": "DzPrzY6aj1hH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.compose import ColumnTransformer\n",
        "\n",
        "num_attribs = [\"Age\", \"SibSp\", \"Parch\", \"Fare\"]\n",
        "cat_attribs = [\"Pclass\", \"Sex\", \"Embarked\"]\n",
        "\n",
        "preprocess_pipeline = ColumnTransformer([\n",
        "                                         (\"num\", num_pipeline, num_attribs),\n",
        "                                         (\"cat\", cat_pipeline, cat_attribs),\n",
        "])"
      ],
      "metadata": {
        "id": "-r1WmvFKj2QB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train = preprocess_pipeline.fit_transform(train_data)\n",
        "X_train"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5hjt2807j3vr",
        "outputId": "dcbe9e13-7620-4478-f061-0e2afbe06746"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[-0.56573582,  0.43279337, -0.47367361, ...,  0.        ,\n",
              "         0.        ,  1.        ],\n",
              "       [ 0.6638609 ,  0.43279337, -0.47367361, ...,  1.        ,\n",
              "         0.        ,  0.        ],\n",
              "       [-0.25833664, -0.4745452 , -0.47367361, ...,  0.        ,\n",
              "         0.        ,  1.        ],\n",
              "       ...,\n",
              "       [-0.10463705,  0.43279337,  2.00893337, ...,  0.        ,\n",
              "         0.        ,  1.        ],\n",
              "       [-0.25833664, -0.4745452 , -0.47367361, ...,  1.        ,\n",
              "         0.        ,  0.        ],\n",
              "       [ 0.20276213, -0.4745452 , -0.47367361, ...,  0.        ,\n",
              "         1.        ,  0.        ]])"
            ]
          },
          "metadata": {},
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_train = train_data[\"Survived\"]"
      ],
      "metadata": {
        "id": "StuBjHTjj46e"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
        "forest_clf.fit(X_train, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pCYf5uzkj6YP",
        "outputId": "47aef4e0-01d4-4b3d-9aa9-9cd2bcdd302c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "RandomForestClassifier(random_state=42)"
            ]
          },
          "metadata": {},
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X_test = preprocess_pipeline.transform(test_data)\n",
        "y_pred = forest_clf.predict(X_test)"
      ],
      "metadata": {
        "id": "vQaDWdOlj7sz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import cross_val_score\n",
        "\n",
        "forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)\n",
        "forest_scores.mean()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5rotzZVwj8uP",
        "outputId": "b44b9d1d-a00f-4e54-e27c-da4847f5c481"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8137578027465668"
            ]
          },
          "metadata": {},
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.svm import SVC\n",
        "\n",
        "svm_clf = SVC(gamma=\"auto\")\n",
        "svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)\n",
        "svm_scores.mean()"
      ],
      "metadata": {
        "id": "YewCo92Wj9pb",
        "outputId": "b8129a54-63c1-44b3-c457-86970ae525ef",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8249313358302123"
            ]
          },
          "metadata": {},
          "execution_count": 25
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(8, 4))\n",
        "plt.plot([1]*10, svm_scores, \".\")\n",
        "plt.plot([2]*10, forest_scores,\".\")\n",
        "plt.boxplot([svm_scores, forest_scores], labels=(\"SVM\", \"Random Forest\"))\n",
        "plt.ylabel(\"Accuracy\")\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "ZxhtmPJhj_Dd",
        "outputId": "d3ba2162-39e4-4f37-f292-a5f82c18e5f5",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 265
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 576x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_data[\"AgeBucket\"] = train_data[\"Age\"] // 15 * 15\n",
        "train_data[[\"AgeBucket\", \"Survived\"]].groupby(['AgeBucket']).mean()"
      ],
      "metadata": {
        "id": "pzZYy9sjkAUH",
        "outputId": "896a759d-5eab-46b7-9b1c-2e905e3e5413",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 268
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "           Survived\n",
              "AgeBucket          \n",
              "0.0        0.576923\n",
              "15.0       0.362745\n",
              "30.0       0.423256\n",
              "45.0       0.404494\n",
              "60.0       0.240000\n",
              "75.0       1.000000"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-a89caf11-26ba-4dd4-98f4-91a56d45ca47\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>AgeBucket</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0.0</th>\n",
              "      <td>0.576923</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>15.0</th>\n",
              "      <td>0.362745</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>30.0</th>\n",
              "      <td>0.423256</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>45.0</th>\n",
              "      <td>0.404494</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>60.0</th>\n",
              "      <td>0.240000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75.0</th>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-a89caf11-26ba-4dd4-98f4-91a56d45ca47')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-a89caf11-26ba-4dd4-98f4-91a56d45ca47 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-a89caf11-26ba-4dd4-98f4-91a56d45ca47');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 27
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train_data[\"RelativesOnboard\"] = train_data[\"SibSp\"] + train_data[\"Parch\"]\n",
        "train_data[[\"RelativesOnboard\", \"Survived\"]].groupby(\n",
        "    ['RelativesOnboard']).mean()"
      ],
      "metadata": {
        "id": "meDWTQTTkBi1",
        "outputId": "fcb802cd-aafe-4a2e-916b-c167c7045c52",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 363
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                  Survived\n",
              "RelativesOnboard          \n",
              "0                 0.303538\n",
              "1                 0.552795\n",
              "2                 0.578431\n",
              "3                 0.724138\n",
              "4                 0.200000\n",
              "5                 0.136364\n",
              "6                 0.333333\n",
              "7                 0.000000\n",
              "10                0.000000"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-5d3cf37b-d290-485b-8242-39da88ed7a77\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Survived</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>RelativesOnboard</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.303538</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.552795</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.578431</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.724138</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0.200000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>0.136364</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>0.333333</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10</th>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-5d3cf37b-d290-485b-8242-39da88ed7a77')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-5d3cf37b-d290-485b-8242-39da88ed7a77 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-5d3cf37b-d290-485b-8242-39da88ed7a77');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 28
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "q21UVyI_kCbr"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
