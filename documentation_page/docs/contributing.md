


To contribute to our work please follow this guide:

## Setup your environment 

```
pip install pip-tools
```
```
pip-sync requirements/production.txt requirements/dev.txt
```


## Integrate new dependencies

To integrate new development dependencies add them to the dev.in file and use the following command to compile them into dev.txt

```
pip-compile -v --output-file requirements/dev.txt requirements/dev.in
```

To integrate new production dependencies add them to the production.in file and use the following command to compile them into production.txt

```
pip-compile -v --output-file requirements/production.txt requirements/production.in
```

For further information please read the documentation of [pip-tools] (https://github.com/jazzband/pip-tools)
 