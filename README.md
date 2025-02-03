# ToyFuji Sim
Personal side quest, convert images to xtran and demosaic. Can apply Fuji film 'recipes' and base curves using CLI. Building up to working on RAW photos for better quality.


## Use:
- Clone this repo
- Navigate into `ToyFujiSim/python/xtraneemu`
  - `pip install -r requirements.txt` to grab the light list of dependancies.
 
- The tool currently functions using CLI arguments, a GUI is planned once I've fleshed out the important stuff to make it more human operable.
  - `python fujifilmemulator.py --help` for basic instructions
  - `python fujifilmemulator.py --list` to see availbile base curves and camera presets
 
  - General usage will look like `python fujifilmemulator.py <photo_path> <preset> <base curve>`
- Once run, tqdm will write status updates as it proccesses the image, the final output will appear in `ToyFujiSim/python/xtranemu/outputs/<curve>_<photoname>_<preset>`

## Discalimer:
- This project is in now way representative of any offical work or research and has nothing to do with fuji film. May very well just be instagram filters.
- Right now it isn't fast, maybe it never will be!

## TODO Items:
- RAW image support
- GUI (Webapp or tkinter)
- Smoother proccess for adding reciepes, auto convert to JSON?
- Implement the Frank Markesteijn algo
- Just make better 
