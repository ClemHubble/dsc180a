# llama_qrc_project

This project focuses on Llama 8B's performance on three mcq datasets (arc easy, arc challenge, openqa) and three open ended datasets (squad-v2, truthfulqa, triviaqa).

## openended_qa

There are three bash scripts to run three python files.  

To evaluate the baseline model, run_qrc_openqa_eval_fast.sh was used to run qrc_openqa_eval_fast.py.  

To evaluate Llama 8B's performance with different temperature and perturbation settings, run_eval_calibrated_qrc.sh was used to run eval_calibrated_qrc.py. Since the job was interrupted, run_calibrated_qrc_resume.sh was used to run eval_calibrated_qrc_resume.py  

The visualizations to summarize the results are stored in october_31.ipynb.  