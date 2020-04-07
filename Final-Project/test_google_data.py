from lib.getDataGoogle import GetGoogleDataset

classesToSelect = ['Box', 'Book', 'Coffee cup']

getData = GetGoogleDataset(debug=False, select_classes=classesToSelect)
# getData.selectClasses()
# getData.saveData()
getData.loadData()
# print(getData.sub_sample_img_url)
print(getData.sub_sample_img_url['Book']['image_url'][0])
getData.downloadImages()
