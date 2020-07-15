from ..adapter import solver, sarsop

sol_opt = {
    'pomdp' : 'package/pomdp-sol/pomdp_sol/proc/volume/example.pomdp',
}

sar_opt = {
    ''      : 'package/pomdp-sol/pomdp_sol/proc/volume/example.pomdp',
    'o'     : 'package/pomdp-sol/pomdp_sol/proc/volume/example.value',
}

sol_out =     'package/pomdp-sol/pomdp_sol/proc/volume/example.solver'
sar_out =     'package/pomdp-sol/pomdp_sol/proc/volume/example.sarsop'

print('Printing output to console ...')
solver(sol_opt)
sarsop(sar_opt)

print('Writing output to file ...')
solver(sol_opt, sol_out)
sarsop(sar_opt, sar_out)
