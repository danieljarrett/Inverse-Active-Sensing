PACK_PGYM = package/pomdp-gym
PACK_PSOL = package/pomdp-sol
TEST_PGDI = package.pomdp-gym.pomdp_gym.envs.testing.diagnosis
TEST_PGDE = package.pomdp-gym.pomdp_gym.envs.testing.decision
TEST_PSOL = package.pomdp-sol.pomdp_sol.proc.testing.simulate
TEST_DECI = testing.decision
TEST_DIAG = testing.diagnosis
TEST_CART = testing.cartpole

DRIVER = driver.py

.PHONY: driver
driver:
	@python -i $(DRIVER)

.PHONY: permit
permit:
	@chmod -R 755 ./

.PHONY: install
install: pack test

.PHONY: clean
clean: purge list

.PHONY: pack
pack: pack_pgym pack_psol

.PHONY: test
test: test_pgde test_pgdi test_psol test_deci  test_diag  test_cart

.PHONY: purge
purge: purge_pack purge_main purge_code

.PHONY: list
list: list_pyth list_ccpp

.PHONY: pack_pgym
pack_pgym:
	@pip install -e $(PACK_PGYM)

.PHONY: pack_psol
pack_psol:
	@pip install -e $(PACK_PSOL)

.PHONY: test_pgde
test_pgde:
	@python -m $(TEST_PGDE)

.PHONY: test_pgdi
test_pgdi:
	@python -m $(TEST_PGDI)

.PHONY: test_psol
test_psol:
	@python -m $(TEST_PSOL)

.PHONY: test_deci
test_deci:
	@python -m $(TEST_DECI)

.PHONY: test_diag
test_diag:
	@python -m $(TEST_DIAG)

.PHONY: test_cart
test_cart:
	@python -m $(TEST_CART)

.PHONY: purge_pack
purge_pack:
	@rm -f volume/*.alpha;   \
	 rm -f volume/*.value;   \
   rm -f volume/*.pg;      \
   rm -f volume/*.solver;  \
   rm -f volume/*.sarsop;  \
   rm -f volume/*.trj.npy; \
   rm -f volume/*.reg.npy;

.PHONY: purge_main
purge_main:
	@rm -f package/pomdp-sol/pomdp_sol/proc/volume/*.alpha;   \
	 rm -f package/pomdp-sol/pomdp_sol/proc/volume/*.value;   \
   rm -f package/pomdp-sol/pomdp_sol/proc/volume/*.pg;      \
   rm -f package/pomdp-sol/pomdp_sol/proc/volume/*.solver;  \
   rm -f package/pomdp-sol/pomdp_sol/proc/volume/*.sarsop;  \
   rm -f package/pomdp-sol/pomdp_sol/proc/volume/*.trj.npy; \
   rm -f package/pomdp-sol/pomdp_sol/proc/volume/*.reg.npy;

.PHONY: purge_code
purge_code:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

.PHONY: list_pyth
list_pyth:
	@find . -name '*.py' | xargs wc -l

.PHONY: list_ccpp
list_ccpp:
	@find . \( -name "*.c" -o -name "*.cpp" \) | xargs wc -l
