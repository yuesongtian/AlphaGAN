import sys
import genotypes
from graphviz import Digraph


def plot(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]
      if j == 0:
        u = "c_{k-2}"
      elif j == 1:
        u = "c_{k-1}"
      else:
        u = str(j-2)
      v = str(i)
      g.edge(u, v, label=op, fillcolor="gray")

  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")

  g.render(filename, view=True)

# gen_dg_loss_g256_d128_searchAutoGFixD_searchWoGumbel_searchWoFixAlphas_searchOnlyUpdateWWorst_searchGWoSkipNoneWithSepconv = Genotype_gen(normal={'1': [('conv_5x5', 0), ('conv_5x5', 1), ('sep_conv_5x5', 2)], '2': [('conv_5x5', 0), ('conv_1x1', 1), ('conv_3x3', 2)], '3': [('conv_1x1', 0), ('conv_5x5', 1), ('sep_conv_3x3', 2)]}, normal_concat=range(2, 6), up={'1': [('deconv', 0), ('bilinear', 1)], '2': [('deconv', 0), ('bilinear', 1)], '3': [('nearest', 0), ('bilinear', 1)]}, up_concat=range(2, 6), skip_2=[('nearest', 0)], skip_2_concat=range(2, 6), skip_3=[('bilinear', 0), ('nearest', 1)], skip_3_concat=range(2, 6))

def plot_gan_Auto(genotype, filename, cell):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("0", fillcolor='darkseagreen2')
  g.node("1", fillcolor='lightblue')
  g.node("2", fillcolor='lightblue')
  g.node("3", fillcolor='lightblue')
  g.node("4", fillcolor='darkseagreen2')


  g.edge('0', '1', label=genotype.up[cell][0][0], fillcolor="gray")
  # g.edge('0', '3', label=genotype.up[cell][1][0], fillcolor='gray')
  
  g.edge('1', '2', label=genotype.normal[cell][0][0], fillcolor='red')
  g.edge('2', '4', label=genotype.normal[cell][1][0], fillcolor='red')
  # g.edge('3', '4', label=genotype.normal[cell][2][0], fillcolor='red')


  g.render(filename, view=True)

def plot_gan_Auto_entire(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node('z', fillcolor='gray')

  g.node("cell1_0", fillcolor='darkseagreen2')
  g.node("cell1_1", fillcolor='lightblue')
  g.node("cell1_2", fillcolor='lightblue')
  g.node("cell1_3", fillcolor='lightblue')
  g.node("cell1_4", fillcolor='darkseagreen2')

  g.node("cell2_0", fillcolor='darkseagreen2')
  g.node("cell2_1", fillcolor='lightblue')
  g.node("cell2_2", fillcolor='lightblue')
  g.node("cell2_3", fillcolor='lightblue')
  g.node("cell2_4", fillcolor='darkseagreen2')

  g.node("cell3_0", fillcolor='darkseagreen2')
  g.node("cell3_1", fillcolor='lightblue')
  g.node("cell3_2", fillcolor='lightblue')
  g.node("cell3_3", fillcolor='lightblue')
  g.node("cell3_4", fillcolor='darkseagreen2')

  g.node('output', fillcolor='red')

  g.edge('z', 'cell1_0', label='fc', fillcolor='blue')

  g.edge('cell1_0', 'cell1_1', label=genotype.up['1'][0][0], fillcolor="gray")
  g.edge('cell1_0', 'cell1_3', label=genotype.up['1'][1][0], fillcolor='gray')
  g.edge('cell1_1', 'cell1_2', label=genotype.normal['1'][0][0], fillcolor='red')
  g.edge('cell1_2', 'cell1_4', label=genotype.normal['1'][1][0], fillcolor='red')
  g.edge('cell1_3', 'cell1_4', label=genotype.normal['1'][2][0], fillcolor='red')

  g.edge('cell2_0', 'cell2_1', label=genotype.up['2'][0][0], fillcolor="gray")
  g.edge('cell2_0', 'cell2_3', label=genotype.up['2'][1][0], fillcolor='gray')
  g.edge('cell2_1', 'cell2_2', label=genotype.normal['2'][0][0], fillcolor='red')
  g.edge('cell2_2', 'cell2_4', label=genotype.normal['2'][1][0], fillcolor='red')
  g.edge('cell2_3', 'cell2_4', label=genotype.normal['2'][2][0], fillcolor='red')
  
  g.edge('cell3_0', 'cell3_1', label=genotype.up['3'][0][0], fillcolor="gray")
  g.edge('cell3_0', 'cell3_3', label=genotype.up['3'][1][0], fillcolor='gray')
  g.edge('cell3_1', 'cell3_2', label=genotype.normal['3'][0][0], fillcolor='red')
  g.edge('cell3_2', 'cell3_4', label=genotype.normal['3'][1][0], fillcolor='red')
  g.edge('cell3_3', 'cell3_4', label=genotype.normal['3'][2][0], fillcolor='red')

  g.edge('cell1_1', 'cell2_4', label=genotype.skip_2[0][0], fillcolor='gray')
  g.edge('cell1_1', 'cell3_4', label=genotype.skip_3[0][0], fillcolor='gray')
  g.edge('cell2_1', 'cell3_4', label=genotype.skip_3[0][0], fillcolor='gray')

  g.edge('cell3_4', 'output', label='to_rgb', fillcolor='green')

  g.render(filename, view=True)

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    sys.exit(1)

  genotype_name = sys.argv[1]
  try:
    genotype = eval('genotypes.{}'.format(genotype_name))
  except AttributeError:
    print("{} is not specified in genotypes.py".format(genotype_name)) 
    sys.exit(1)

  plot_gan_Auto(genotype, 'cell_small_1', "1")
  # plot_gan_Auto(genotype, 'cell_small_2', "2")
  # plot_gan_Auto(genotype, 'cell_small_3', "3")

  # plot_gan_Auto_entire(genotype, 'GANAS_large_step')

