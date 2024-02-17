import xlwt

book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('sheet1', cell_overwrite_ok=True)

# col = ('电影详情链接','图片链接')
#
# #加载将要导入的excel文件
# for i in range(0,8):
#     sheet.write(0,i,col[i])

datalist = ['www','www图片','西游记','xiyouji','100分','0人','很好','超级棒','www2','www图片2','西游记2','xiyouji2','1000分','1人','很棒','一级棒']
len = len(datalist)

for i in range(0,len):
    sheet.write(i,1,datalist[i])

savepath = 'excel_nuclues.xls'
book.save(savepath)