# 코드 실행 안내

수집·분류 코드는 `crawling/`, 분석 코드는 `analysis/`, 결과 검토용 노트북은 `notebooks/`에 있다. 연구 입력과 중간 산출물은 포함하지 않으며, 공개 코드만으로 결과 재현을 검증하지 않았다. 패키지 버전도 고정하지 않았으므로 설치 환경에 따라 호환성 확인이 필요하다.

## 준비

Python 환경에서 `pip install -r code/requirements.txt`로 패키지를 설치한다. 전체어휘 LDA의 한국어 형태소 분석에는 Java와 KoNLPy가 필요하다. Java가 자동으로 감지되지 않는 경우 설치 환경에 맞게 `JAVA_HOME`을 설정한다. 수집 시 PDF OCR에는 Tesseract 실행 프로그램과 필요한 언어 데이터, 동적 페이지에는 Chrome 및 Selenium에서 사용할 드라이버가 필요하다. Chrome 자동 탐색이 되지 않으면 `CHROME_BINARY`에 실행 파일 경로를 지정한다. 노트북은 Jupyter에서 연다.

분석에 필요한 비공개 연구자료의 최상위 폴더를 `ESG_RESEARCH_ROOT`로 지정한다. 다음 경로는 그 폴더를 기준으로 한다.

- `comp/0_crawling/phase5_page_type/esg_final_dataset_v3.xlsx`: 최종 데이터
- `comp/0_crawling/phase5_page_type/저신뢰플래그_수동검토_v3.xlsx`: 수동 검토표
- `comp/file/lists/univ_list.xlsx`: 대학·학과 목록
- `korea/preprocessing/esg_concepts_kr.json`, `japan/preprocessing/esg_concepts_ja.json`: 한일 ESG 사전
- `japan/preprocessing/admin_stopwords_ja.txt`: 일본어 불용어

수집·분류 단계에는 시드 목록, 페이지 목록과 수동 라벨 등 별도의 단계별 입력이 필요하다. 분류 API 호출에는 `OPENAI_API_KEY`가 필요하며 호출 비용이 발생할 수 있다. 코드별 경로와 입력은 각 스크립트와 노트북에서 확인한다.

한글·일본어 그래프와 워드클라우드에 사용할 폰트 파일은 각각 `ESG_FONT_KR`, `ESG_FONT_JP`로 지정할 수 있다. 미지정 시 기존 macOS 한국어 폰트와 연구자료 내 일본어 폰트 경로를 사용한다. 폰트 파일도 포함하지 않는다.

## 실행 예시

프로젝트 루트에서 필요한 입력을 준비하고 환경변수를 설정한 후 실행한다. 분석 결과는 연구자료 폴더의 `comp/analysis/output/`에 저장된다.

```sh
export ESG_RESEARCH_ROOT="/path/to/research-data"
python code/analysis/run_analysis_pipeline.py
python code/analysis/export_comparison_workbook.py
jupyter notebook
```

수집과 분류는 네트워크 및 외부 서비스에 접근하고 파일을 생성한다. 실행 전에 각 단계의 입력과 설정을 확인한다.
