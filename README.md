# blather

Train and use generative text models in a few lines of code.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install blather. 
```bash
pip install blather
```

## Usage

```python
from blather import Blather

blather = Blather()

# fine tunes an appropriate model on your dataset
blather.read(['Text Examples One', 'Second Text Example'... 'Text Example k'])

# returns a text sample generated from the model
blather.write('Sample text to complete')

# saves model
blather.save('model.pt')

# load model from previous training
blather.load('model.pt')
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache](https://choosealicense.com/licenses/apache-2.0/)
