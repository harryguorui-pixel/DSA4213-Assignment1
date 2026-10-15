part 2:
python -m tests.test_part2_setup
python -m src.part2.train_skipgram
python -m src.part2.prepare_glove
python run_part2.py --setting B
python run_part2.py --setting C
python evaluate_part2.py
python generate_part2.py
python plot_part2.py