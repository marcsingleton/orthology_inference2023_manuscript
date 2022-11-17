"""Plot number of annotated genomes by NCBI."""

import os

import matplotlib.pyplot as plt

data = [['2001', 1],
        ['2002', 2],
        ['2003', 3],
        ['2004', 6],
        ['2005', 10],
        ['2006', 12],
        ['2007', 16],
        ['2008', 19],
        ['2009', 22],
        ['2010', 28],
        ['2011', 40],
        ['2012', 53],
        ['2013', 109],
        ['2014', 203],
        ['2015', 276],
        ['2016', 361],
        ['2017', 435],
        ['2018', 506],
        ['2019', 582],
        ['2020', 708],
        ['2021', 820],
        ['2022', 923]]

if not os.path.exists('out/'):
    os.mkdir('out/')

xs, ys = zip(*data)
fig, ax = plt.subplots(layout='constrained')
ax.bar(xs, ys)
ax.tick_params('x', rotation=60)
ax.set_xlabel('Year')
ax.set_ylabel('Cumulative number of annotated genomes')
fig.savefig('out/ncbi.png', dpi=300)
fig.savefig('out/ncbi.tiff', dpi=300)
plt.close()
