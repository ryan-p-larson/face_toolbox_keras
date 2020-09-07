import setuptools

setuptools.setup(
  name='face_toolbox',
  version='0.1dev0',
  author='Ryan Larson',
  author_email='ryalarson11@gmail.com',
  description='Pre-packaged Keras nets for face/eye parsing, etc.',
  long_description=open('README.md').read(),
  packages=setuptools.find_packages(),
  install_requires=['numpy', 'opencv-python', 'keras', 'tensorflow', 'shapely', 'scikit-image'],
  include_package_data=True,
  package_data={
    '': ['models/parser/BiSeNet/BiSeNet_keras.h5', '*.h5', '*.npy', '*.json']
  },
  python_requires='>=3.6'
)