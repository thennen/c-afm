# df = pd.read_pickle whatever

iv = df[df.type == 'iv']
riv = iv[iv.folder == '09-Jun-2017'].sort('ctime')

# Overlap all the iv loops.  At least color them according to something
fig, ax = plt.subplots()
colors = iter(['#F60202', '#019494', '#F67102', '#02C502'])
labels = iter(['Conducting region before','Insulating region before','Conducting region after','Insulating region after'])
for loc, group in riv.groupby('x_offset'):
    color = next(colors)
    for row, iv in group.iterrows():
        plot(iv.V, iv.I * 1e9, color=color)
        #plot(iv.V2, iv.I2 * 1e9, color=color)
    ax.lines[-1].set_label(next(labels))
legend()
ylabel('Tip current [nA]')
xlabel('Applied Voltage [V]')

fig.savefig('C:\\t\\LCAFM\\09-Jun-2017\\ivdata\\all_overlap.png')


# Overlap the repeated iv loops
fig, ax = plt.subplots()
colors = iter(['#F60202', '#019494', '#F67102', '#02C502'])
labels = iter(['Conducting region before','Insulating region before','Conducting region after','Insulating region after'])
for loc, group in riv.groupby('x_offset'):
    color = next(colors)
    for row, iv in group.iterrows():
        plot(iv.V, iv.I * 1e9, color=color)
        #plot(iv.V2, iv.I2 * 1e9, color=color)
    ax.lines[-1].set_label(next(labels))
    ylabel('Tip current [nA]')
    xlabel('Applied Voltage [V]')
    legend()
    fig.savefig('C:\\t\\LCAFM\\09-Jun-2017\\ivdata\\overlap_{}.png'.format(group.run.iloc[0]))
    ax.cla()

# IV loops are easy to average because they have the same x coordinates ..
def meanloop(group):
    # take average of forward and backward loops
    # Wait there are no backward loops this time..
    I1 = group.I.mean()
    V1 = group.V.mean()
    #I2 = group.I2.mean()
    #V2 = group.V2.mean()
    #meanI = np.mean(I1, I2)
    #meanV = np.mean(V1, V2)
    return pd.Series(dict(I = I1, V=V1))
iv_avg = riv.groupby('x_offset').apply(meanloop)

fig, ax = plt.subplots()
colors = iter(['#F60202', '#019494', '#F67102', '#02C502'])
labels = iter(['Conducting region before','Insulating region before','Conducting region after','Insulating region after'])
for loc, loop in iv_avg.iterrows():
    color = next(colors)
    plot(loop.V, loop.I * 1e9, color=color)
    ax.lines[-1].set_label(next(labels))
legend()
ylabel('Tip current [nA]')
xlabel('Applied Voltage [V]')
fig.savefig('C:\\t\\LCAFM\\09-Jun-2017\\ivdata\\averaged.png')

##  Plot each individually
colors = iter(['#F60202', '#019494', '#F67102', '#02C502'])
labels = iter(['Conducting region before','Insulating region before','Conducting region after','Insulating region after'])
fig, ax = plt.subplots()
for loc, group in riv.groupby('x_offset'):
    color = next(colors)
    label = next(labels)
    for row, iv in group.iterrows():
        plot(iv.V, iv.I * 1e9, color=color, label=label)
        #plot(iv.V2, iv.I2 * 1e9, color=color)
        legend()
        ylabel('Tip current [nA]')
        xlabel('Applied Voltage [V]')
        fig.savefig('C:\\t\\LCAFM\\09-Jun-2017\\ivdata\\{}.png'.format(iv.id))
        ax.cla()


## export iv to xls for jonathan.

# Metadata
riv.cycle = np.int16(riv.cycle)
riv.run = np.int16(riv.run)
riv.mystery_id = np.int16(riv.mystery_id)
columns = ['sample_name', 'mystery_id', 'run', 'cycle', 'folder', 'filename', 'x_offset','y_offset']
riv[columns].sort_values(['mystery_id', 'run', 'cycle']).to_excel('C:\\t\\LCAFM\\09-Jun-2017\\ivdata\\metadata.xls', index=False)

for ind, iv in riv.iterrows():
    with open('C:\\t\\LCAFM\\09-Jun-2017\\ivdata\\{}.csv'.format(iv.id), 'w') as f:
        header = iv[columns]
        for k,v in header.iteritems():
            f.write('#{},{}\n'.format(k, v))
        singleloop = pd.DataFrame(dict(I=iv.I, V=iv.V))
        singleloop.to_csv(f, sep=',', index=False)
